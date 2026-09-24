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
from .db import SessionRecord, Task, TaskCluster, Utterance, now_iso, tenant_session
from .engine.base import Engine, normalize
from .speakers import bump_version, match_or_create

log = logging.getLogger(__name__)

FFMPEG_TIMEOUT = 120

MIN_NEW_SPEAKER_SECONDS = 1.0  # 亚秒级碎片的音色向量不可靠，只允许并入已有说话人

AUTO_MERGE_SIM = 0.62         # 库自合并：薄卡并入厚卡的相似度线
AUTO_MERGE_MAX_VECTORS = 11   # 只清理碎片：向量数不超过它的卡才算薄卡。
                              # 两张确立的厚卡即使相似也不自动并——近场相近
                              # 异人（实测 0.696）与远场同代碎片（0.64-0.67）
                              # 在绝对相似度上无法区分，只能按"谁像碎片"分治，
                              # 厚卡合并留给人工（memoclip 有合并 UI）
AUTO_MERGE_DORMANT_CHUNKS = 10  # 休眠判定：连续这么多片没有身份确认的薄卡才算
                              # 碎片（只认建卡/整权合并；挂靠不增加向量数却会归
                              # utterance，靠它判活跃会让僵尸卡永远清醒）。
AUTO_MERGE_MAX_SPOKEN_MS = 60_000  # 说话时长上限：超过它的卡不是碎片——远场
                              # 少言者说话次数少但单次长，向量数很低却有大量语音
                              # （实测 168s 的卡只有几个向量被误并，用户真值标注
                              # 后 DER 0.163→0.251 的负优化）。

REATTRIBUTE_MARGIN = 0.10     # 重归属改判余量：新卡须比现归属像这么多才改写。


def reattribute_speakers(db, session_id: str, threshold: float) -> int:
    """用当前声纹库重判历史簇归属，返回改判的簇数（跟在库收敛之后跑）。

    远场少言者每次说话只有 1-4 秒、单片攒不够建卡素材，其声纹卡往往在
    录音后段才出现——早前的话全部被灰区挂靠吸进了别人（实测 E 的 47 秒
    散落 6 张卡，纯度 0.30）。process_task 把每簇音色向量落 task_clusters，
    这里在新卡就位后回头重新裁决：新归属明显更像（余量 REATTRIBUTE_MARGIN）
    才改写簇与其 utterances 的 person_id。改判同时回填认人元数据——过阈值
    才算确认归并，否则仍按灰区挂靠报，并置 reattributed 标记。"""
    from .db import TaskCluster, Utterance
    from .speakers import load_persons, resolve_root
    persons = load_persons(db, session_id)
    cen = {p.id: normalize(np.frombuffer(p.centroid, dtype=np.float32))
           for p in persons.values() if not p.merged_into and p.centroid}
    if not cen:
        return 0
    rows = db.scalars(select(TaskCluster)
                      .where(TaskCluster.session_id == session_id)).all()
    changed = 0
    for r in rows:
        cur = resolve_root(persons, r.person_id).id
        if cur not in cen:
            continue
        emb = normalize(np.frombuffer(r.emb, dtype=np.float32))
        best_pid, best_sim, second_sim = None, -1.0, -1.0
        for pid, c in cen.items():
            sim = float(emb @ c)
            if sim > best_sim:
                second_sim, best_sim, best_pid = best_sim, sim, pid
            elif sim > second_sim:
                second_sim = sim
        if best_pid != cur and best_sim - float(emb @ cen[cur]) >= REATTRIBUTE_MARGIN:
            r.person_id = best_pid
            r.action = "merge" if best_sim >= threshold else "attach"
            r.best_sim = best_sim
            r.second_sim = second_sim
            r.reattributed = True
            db.execute(update(Utterance).where(
                Utterance.task_id == r.task_id,
                Utterance.seq.in_(r.seqs)).values(person_id=best_pid))
            changed += 1
    return changed


def consolidate_speakers(db, session_id: str) -> int:
    """收敛会话声纹库：把休眠的碎片薄卡并回最像的厚卡，返回合并次数。

    远场相近嗓子的片级相似度长期落在灰区，同一人会被拆出多张卡（实测
    一晚真实使用攒出 11 张）。这里把"向量数少 + 说话时长少 + 已连续数片
    没有身份确认"的休眠碎片并回质心最像的厚卡，沿用人工合并路径（保留
    人工命名约束、审计与版本联动）。活跃成长中的卡、说话多的卡与厚卡
    之间永不自动并。"""
    from .db import Task, Utterance
    from .speakers import load_persons, merge as merge_persons
    cur_seq = db.scalar(select(func.max(Task.seq)).where(
        Task.session_id == session_id))
    if cur_seq is None:
        return 0
    spoken = dict(db.execute(
        select(Utterance.person_id, func.sum(Utterance.end_ms - Utterance.start_ms))
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
            if (spoken.get(b.id) or 0) > AUTO_MERGE_MAX_SPOKEN_MS:
                continue  # 说话多的卡不是碎片（向量少≠素材少）
            if cur_seq - (b.last_merge_seq or 0) < AUTO_MERGE_DORMANT_CHUNKS:
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
        # 错字——同音/近音两档，保守取向见 app/hotwords.py。
        if task.hotwords:
            for seg in segments:
                seg.text = apply_hotwords(seg.text, task.hotwords)
        segments.sort(key=lambda s: s.start_ms)

        # 音频内聚类合并 FunASR 的碎编号，再与会话说话人库对齐
        clusters = cluster_local_speakers(
            lambda spans: engine.embed(wav, spans), segments, cfg.speaker_threshold)

        with tenant_session(db_engine, tenant) as db:
            spk_to_person: dict[int, str] = {}
            spk_to_cluster = {spk: c for c in clusters for spk in c["spks"]}
            for c in clusters:
                m = match_or_create(db, session_id, c["emb"], cfg.speaker_threshold,
                                    c["ms"], cfg.speaker_min_enroll_ms,
                                    merge_seq=task.seq)
                c["match"] = m
                for spk in c["spks"]:
                    spk_to_person[spk] = m.pid
            seqs_by_cluster = {id(c): [] for c in clusters}
            for i, s in enumerate(segments, start=1):
                db.add(Utterance(task_id=task_id, session_id=session_id, seq=i,
                                 start_ms=s.start_ms, end_ms=s.end_ms, text=s.text,
                                 person_id=spk_to_person[s.local_spk]))
                seqs_by_cluster[id(spk_to_cluster[s.local_spk])].append(i)
            for c in clusters:  # 簇音色向量落库，供录音潮后的重归属回头改判
                m = c["match"]
                db.add(TaskCluster(task_id=task_id, session_id=session_id,
                                   person_id=m.pid, ms=c["ms"],
                                   emb=np.asarray(c["emb"], dtype=np.float32).tobytes(),
                                   seqs=seqs_by_cluster[id(c)],
                                   action=m.action, best_sim=m.best_sim,
                                   second_sim=m.second_sim))
            elapsed_ms = int((time.monotonic() - started) * 1000)
            done_task = db.get(Task, task_id)
            done_task.status = "done"
            done_task.error = None
            done_task.duration_ms = elapsed_ms
            done_task.done_at = now_iso()
            done_task.audio_path = None
            done_task.hotwords = None  # 热词已消费完；人物名明文不留库（随任务清除）
            db.execute(update(SessionRecord)
                       .where(SessionRecord.id == session_id)
                       .values(last_active_at=now_iso()))
            # 录音潮结束（本会话已无排队/处理中任务）后收敛声纹库：
            # 先并休眠碎片，再用最终库对历史挂靠簇重归属（少言者建卡晚，
            # 早前被灰区吸走的话在新卡就位后改判回来）
            pending = db.scalar(select(func.count()).select_from(Task).where(
                Task.session_id == session_id,
                Task.status.in_(["queued", "processing"])))
            if not pending:
                merged = consolidate_speakers(db, session_id)
                moved = reattribute_speakers(db, session_id, cfg.speaker_threshold)
                if merged or moved:
                    bump_version(db, session_id)
                    log.info("声纹库收敛 tenant=%s session=%s 合并=%d 改判=%d",
                             tenant, session_id, merged, moved)
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
