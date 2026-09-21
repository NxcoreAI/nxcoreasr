"""转写流水线：转码 → 转写 → 认人 → 落库 → 清理音频。

失败时保留原始音频以便 retry；日志不打转写文本与人名。
"""
import logging
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
from sqlalchemy import update

from .config import AppConfig
from .hotwords import apply_hotwords
from .db import SessionRecord, Task, Utterance, now_iso, tenant_session
from .engine.base import Engine, normalize
from .speakers import match_or_create

log = logging.getLogger(__name__)

FFMPEG_TIMEOUT = 120

MIN_NEW_SPEAKER_SECONDS = 1.0  # 亚秒级碎片的音色向量不可靠，只允许并入已有说话人


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
        # 转写成功即删原始音频；失败保留供 retry（本地下载件统一在 finally 清）
        if downloaded:
            storage.delete(key)
        else:
            Path(src).unlink(missing_ok=True)
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
        log.warning("任务失败 tenant=%s task=%s err=%s", tenant, task_id, str(e)[:200])
    finally:
        Path(wav).unlink(missing_ok=True)
        if downloaded:
            Path(src).unlink(missing_ok=True)
