"""转写流水线：转码 → 转写 → 认人 → 落库 → 清理音频。

失败时保留原始音频以便 retry；日志不打转写文本与人名。
"""
import logging
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
from sqlalchemy import func, select, update

from .config import AppConfig
from . import events
from .hotwords import apply_hotwords
from .db import SessionRecord, Task, Utterance, now_iso, tenant_session
from .engine.base import Engine, normalize
from .speakers import match_or_create

log = logging.getLogger(__name__)

FFMPEG_TIMEOUT = 120

MIN_NEW_SPEAKER_SECONDS = 1.0  # 亚秒级碎片的音色向量不可靠，只允许并入已有说话人

AUTO_MERGE_SIM = 0.62         # 库自合并：薄卡并入厚卡的相似度线
AUTO_MERGE_MAX_VECTORS = 11   # 只清理碎片：向量数不超过它的卡才算薄卡。
                              # 两张确立的厚卡即使相似也不自动并——近场相近
                              # 异人（实测 0.696）与远场同代碎片（0.64-0.67）
                              # 在绝对相似度上无法区分，只能按"谁像碎片"分治，
                              # 厚卡合并留给人工（memoclip 有合并 UI）
AUTO_MERGE_DORMANT_CHUNKS = 10  # 休眠判定：连续这么多片没再说话的薄卡才算碎片。
                              # 合法新卡在成长期（向量数≤上限）也会短暂落进薄卡
                              # 区间，若与相近异人卡相似度达标会被误回收；成长中的
                              # 卡被密集触达不会休眠，碎片卡是偶发的长间隔（实测
                              # 远场卫星卡 95 片里只出现 3-11 次，主导卡过半）。


def consolidate_speakers(db, session_id: str) -> int:
    """收敛会话声纹库：把休眠的碎片薄卡并回最像的厚卡，返回合并次数。

    远场相近嗓子的片级相似度长期落在灰区，同一人会被拆出多张卡（实测
    一晚真实使用攒出 11 张）。这里把"向量数少 + 已连续数片不再说话"的
    休眠碎片并回质心最像的厚卡，沿用人工合并路径（保留人工命名约束、
    审计与版本联动）。活跃成长中的卡与厚卡之间永不自动并。"""
    from sqlalchemy import func as sa_func

    from .db import Task, Utterance
    from .speakers import load_persons, merge as merge_persons
    cur_seq = db.scalar(select(sa_func.max(Task.seq)).where(
        Task.session_id == session_id))
    if cur_seq is None:
        return 0
    last_seq = dict(db.execute(
        select(Utterance.person_id, sa_func.max(Task.seq))
        .join(Task, Task.id == Utterance.task_id)
        .where(Utterance.session_id == session_id)
        .group_by(Utterance.person_id)).all())
    rounds = 0
    while True:
        persons = load_persons(db, session_id)
        roots = [p for p in persons.values() if not p.merged_into and p.centroid]
        cen = {p.id: normalize(np.frombuffer(p.centroid, dtype=np.float32))
               for p in roots}
        best = None  # (sim, from_id, into_id)
        for b in roots:
            if b.embed_count > AUTO_MERGE_MAX_VECTORS:
                continue  # 只有薄卡能被并
            if cur_seq - (last_seq.get(b.id) or 0) < AUTO_MERGE_DORMANT_CHUNKS:
                continue  # 还在活跃成长，不是碎片
            cands = [a for a in roots
                     if a.id != b.id and a.embed_count > b.embed_count]
            for a in cands:
                sim = float(cen[a.id] @ cen[b.id])
                if sim >= AUTO_MERGE_SIM and (best is None or sim > best[0]):
                    best = (sim, b.id, a.id)
        if best is None:
            return rounds
        try:
            merge_persons(db, session_id, best[1], best[2])
            rounds += 1
        except Exception:
            return rounds  # 命名冲突等留给人工；该薄卡不再尝试


class AudioError(Exception):
    pass


def transcode(src: str, dst: str) -> None:
    """任意输入 → 16k 单声道 PCM16 wav。"""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", src, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", dst],
        capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
    )
    if proc.returncode != 0:
        raise AudioError(f"音频无法解码: {proc.stderr.strip()[:200]}")


def cluster_local_speakers(embed_fn, segments, threshold: float) -> list[dict]:
    """音频内先聚类：FunASR 的 spk 编号在长音频里会把同一人拆成多个编号，
    先用音色向量在本音频内合并；短碎片段只并入已有簇，绝不独立成新说话人。

    embed_fn(spans) -> 归一化音色向量；线上传 engine 的封装，
    diarize_eval 重放时传缓存（换阈值不重跑 cam++）。
    """
    spans_by_spk: dict[int, list[tuple[int, int]]] = {}
    for s in segments:
        spans_by_spk.setdefault(s.local_spk, []).append((s.start_ms, s.end_ms))
    if not spans_by_spk:
        return []

    clusters = []
    for spk, spans in spans_by_spk.items():
        ms = sum(e - b for b, e in spans)
        clusters.append({"emb": embed_fn(spans), "ms": ms, "spks": [spk]})
    clusters.sort(key=lambda c: c["ms"], reverse=True)  # 素材多的先定簇，质心更稳

    min_ms = MIN_NEW_SPEAKER_SECONDS * 1000
    # 素材全都不足时保底用最多的一簇当锚点，保证短音频也能建出第一个人
    anchor_ms = [c for c in clusters if c["ms"] >= min_ms] or clusters[:1]
    anchor_ids = {id(c) for c in anchor_ms}
    floats = [c for c in clusters if id(c) not in anchor_ids]

    # 合并到收敛：第一轮并入会改变簇质心，A、C 擦边但经 B 中转本应合并的
    # 情况只有再过一轮才能发现（簇数单调下降，必然终止）
    while True:
        merged: list[dict] = []
        for c in anchor_ms:
            best, best_sim = None, -1.0
            for m in merged:
                sim = float(np.dot(c["emb"], m["emb"]))
                if sim > best_sim:
                    best, best_sim = m, sim
            if best is not None and best_sim >= threshold:
                _absorb(best, c)
            else:
                merged.append(c)
        if len(merged) == len(anchor_ms):
            break
        anchor_ms = merged
    merged = anchor_ms
    for c in floats:
        target = max(merged, key=lambda m: float(np.dot(c["emb"], m["emb"])))
        _absorb(target, c)
    return merged


def _absorb(target: dict, frag: dict) -> None:
    target["emb"] = normalize(target["emb"] * target["ms"] + frag["emb"] * frag["ms"])
    target["ms"] += frag["ms"]
    target["spks"].extend(frag["spks"])


def process_task(cfg: AppConfig, engine: Engine, db_engine, storage, tenant: str,
                 task_id: str) -> None:
    with tenant_session(db_engine, tenant) as db:
        task = db.get(Task, task_id)
        if task is None or task.status != "queued":
            return
        session = db.get(SessionRecord, task.session_id)
        if session is None or session.status != "active":
            return  # 会话已在删除中，删除流程会清掉任务
        task.status = "processing"
        session_id = task.session_id
        key = task.audio_path

    src = storage.materialize(key)  # OSS：先下到本地临时；本地盘：原路径
    downloaded = src != key
    wav = src + ".16k.wav"
    started = time.monotonic()
    try:
        t = time.monotonic()
        transcode(src, wav)
        transcode_ms = int((time.monotonic() - t) * 1000)
        with wave.open(wav, "rb") as w:
            record_ms = int(w.getnframes() / w.getframerate() * 1000)
        t = time.monotonic()
        segments = engine.transcribe(wav)
        transcribe_ms = int((time.monotonic() - t) * 1000)
        # 热词同音纠偏（用户人物库专有名词，任务级携带）：落库前修正专有名词
        # 错字——严格同音替换，保守取向见 app/hotwords.py。
        if task.hotwords:
            for seg in segments:
                seg.text = apply_hotwords(seg.text, task.hotwords)
        segments.sort(key=lambda s: s.start_ms)

        # 音频内聚类合并 FunASR 的碎编号，再与会话说话人库对齐
        clusters = cluster_local_speakers(
            lambda spans: engine.embed(wav, spans), segments, cfg.speaker_threshold)

        with tenant_session(db_engine, tenant) as db:
            spk_to_person: dict[int, str] = {}
            for c in clusters:
                pid = match_or_create(db, session_id, c["emb"], cfg.speaker_threshold,
                                      c["ms"], cfg.speaker_min_enroll_ms)
                for spk in c["spks"]:
                    spk_to_person[spk] = pid
            for i, s in enumerate(segments, start=1):
                db.add(Utterance(task_id=task_id, session_id=session_id, seq=i,
                                 start_ms=s.start_ms, end_ms=s.end_ms, text=s.text,
                                 person_id=spk_to_person[s.local_spk]))
            elapsed_ms = int((time.monotonic() - started) * 1000)
            done_task = db.get(Task, task_id)
            done_task.status = "done"
            done_task.error = None
            done_task.duration_ms = elapsed_ms
            done_task.done_at = now_iso()
            done_task.audio_path = None
            db.execute(update(SessionRecord)
                       .where(SessionRecord.id == session_id)
                       .values(last_active_at=now_iso()))
            # 录音潮结束（本会话已无排队/处理中任务）后收敛声纹库：
            # 合并远场灰区攒出的同代碎片卡
            pending = db.scalar(select(func.count()).select_from(Task).where(
                Task.session_id == session_id,
                Task.status.in_(["queued", "processing"])))
            if not pending:
                merged = consolidate_speakers(db, session_id)
                if merged:
                    log.info("声纹库收敛 tenant=%s session=%s 合并=%d",
                             tenant, session_id, merged)
        # 转写成功即删原始音频；失败保留供 retry（本地下载件统一在 finally 清）
        if downloaded:
            storage.delete(key)
        else:
            Path(src).unlink(missing_ok=True)
        events.publish(tenant, task_id, "done")
        speech_ms = sum(s.end_ms - s.start_ms for s in segments)
        speech_pct = speech_ms * 100 // record_ms if record_ms else 0
        log.info("任务完成 tenant=%s task=%s 耗时=%dms 句数=%d 录音=%.1fs 语音=%.1fs(%d%%) 转码=%dms 转写=%dms",
                 tenant, task_id, elapsed_ms, len(segments),
                 record_ms / 1000, speech_ms / 1000, speech_pct,
                 transcode_ms, transcribe_ms)
    except Exception as e:
        # 保留原始音频供 retry
        with tenant_session(db_engine, tenant) as db:
            failed = db.get(Task, task_id)
            if failed is not None:
                failed.status = "failed"
                failed.error = str(e)[:500]
                failed.done_at = now_iso()
                events.publish(tenant, task_id, "failed")
        log.warning("任务失败 tenant=%s task=%s err=%s", tenant, task_id, str(e)[:200])
    finally:
        Path(wav).unlink(missing_ok=True)
        if downloaded:
            Path(src).unlink(missing_ok=True)
