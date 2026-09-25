#!/usr/bin/env python
"""day-split 场景灌数：过去时间轴 + 场间真实静默间隙 + 预建日程窗口。

用法：
  .venv/bin/python push_day_split_scenario.py cal     # 日程变体：5 日历（含 1 no-show 静默窗）
  .venv/bin/python push_day_split_scenario.py mixed   # 混合变体：2 场有日程 + 2 场间隙对话

时间轴锚在过去（anchor = now - 总跨度 - 2min），满速推送不配速——day-split 的
日历匹配只要求与分段跨度重叠，过去/未来均可。间隙期间不推任何分段（转写行
静默 ≥180s 供 silence_gap 锚点命中）。
"""
import hashlib
import json
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import wave

from push_case_to_memoclip import call

CASES = Path("/Users/nxcore/Git/nxcoreasr/diarize_cases")
# 连灌模式（--burst）：满速上传不等转写——会占满本地 ASR，真机排队饿死，慎用
BURST = "--burst" in sys.argv

# 场景：窗口列表 = (case 目录名, chunk 数, 日程标题或 None, 前置间隙秒)
# 日程标题 None 且音频存在 → 间隙对话；case None → 纯静默 no-show 窗口。
SCENARIOS = {
    "cal": {
        "title": "全天拆分实测·日程变体（5 窗含 no-show）",
        "windows": [
            (None, 0, None, 0),
            ("real-meeting-4p-20260922", 40, "智能质检升级方案评审会", 240),
            ("three-person-chat-20260921", 30, "三人日常碰头", 240),
            (None, 0, "客户现场勘察", 240),          # no-show：有日程无音频
            ("real-meeting-6p-20260922", 40, "六人产研例会", 240),
            ("real-meeting-7p-20260922", 40, "七人跨部门对齐", 240),
        ],
    },
    "mixed": {
        "title": "全天拆分实测·混合变体（2 日程 + 2 间隙）",
        "windows": [
            (None, 0, None, 0),
            ("real-meeting-5p-20260922", 40, "AISHELL 五人评审", 210),
            ("real-meeting-3p-20260922", 40, None, 210),
            ("four-person-synth-20260921", 40, "四人方案对齐", 210),
            ("real-meeting-6p-20260922", 40, None, 210),
        ],
    },
}


def chunk_ms(case: Path, i: int) -> int:
    with wave.open(str(case / f"cache/{i:02d}.wav"), "rb") as w:
        return int(w.getnframes() / w.getframerate() * 1000)


def push_chunk_nopoll(event_id, case: Path, i, began_at, dur_ms):
    """上传一片并返回 job id（不等转写完成——末尾统一轮询）。

    注意：满速连灌会瞬间积压上百个 job，把本地 llama-funasr 打满，
    真机设备的分段会排队饿死（2026-09-24 实测用户报障）。默认串行
    （--burst 才连灌）；时间轴是合成锚定，不需要实时满速。"""
    raw = (case / f"chunks/{i:02d}.m4a").read_bytes()
    key = f"daysplit-{case.name[:12]}-{i:02d}-{uuid.uuid4().hex[:10]}"
    seg = call("POST", f"/events/{event_id}/recording-start", {
        "sourceKind": "phone", "idempotencyKey": key,
        "beganAt": began_at.isoformat().replace("+00:00", "Z")})["data"]
    seg_id = seg["segment"]["id"] if "segment" in seg else seg["id"]
    call("POST", f"/events/{event_id}/segments/{seg_id}/complete",
         {"endedAt": (began_at + timedelta(milliseconds=dur_ms)).isoformat().replace("+00:00", "Z")})
    job = call("POST", "/asr-jobs", {
        "eventId": event_id, "segmentId": seg_id,
        "fileName": f"chunk-{i:02d}.m4a", "mimeType": "audio/mp4",
        "fileSize": len(raw), "contentHash": hashlib.sha256(raw).hexdigest(),
        "estimatedDurationMs": dur_ms,
        "idempotencyKey": key, "diarizationEnabled": True})["data"]
    auth = call("POST", f"/asr-jobs/{job['id']}/upload-authorization")["data"]
    up = urllib.request.Request(auth["uploadUrl"], data=raw, method="PUT", headers=auth["headers"])
    urllib.request.urlopen(up, timeout=300).read()
    call("POST", f"/asr-jobs/{job['id']}/upload-complete", {"objectKey": auth["objectKey"]})
    return job["id"]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SCENARIOS:
        print(__doc__)
        sys.exit(2)
    sc = SCENARIOS[sys.argv[1]]

    # 计算总跨度（音频时长 + 间隙），锚到过去
    def window_len(win):
        case, n, _, _ = win
        if case is None:
            return 240_000  # no-show 静默窗按 4 分钟
        return sum(chunk_ms(CASES / case, i) for i in range(1, n + 1))
    # 首行是占位（无前置间隙）；后续每行自带前置间隙
    span_ms = 0
    for idx, win in enumerate(sc["windows"][1:], 1):
        span_ms += win[3] * 1000
        span_ms += window_len(win)
    anchor = datetime.now(timezone.utc) - timedelta(milliseconds=span_ms) - timedelta(minutes=2)

    # 预建日程（窗口起点先按音频时长排出来）
    cursor = anchor
    plans = []  # (win, window_start, window_end_ms_len)
    for win in sc["windows"][1:]:
        cursor += timedelta(seconds=win[3])
        start = cursor
        length = window_len(win)
        cursor += timedelta(milliseconds=length)
        plans.append((win, start, length))
        if win[2]:
            cal = call("POST", "/calendar/events", {
                "title": win[2],
                "startsAt": start.isoformat().replace("+00:00", "Z"),
                "endsAt": cursor.isoformat().replace("+00:00", "Z"),
                "reminderMinutes": 0})["data"]
            print(f"CALENDAR {cal['id']} 「{win[2]}」 {start.strftime('%H:%M')} → {cursor.strftime('%H:%M')}", flush=True)

    ev = call("POST", "/events", {"title": sc["title"]})["data"]
    event_id = ev["id"]
    print(f"EVENT {event_id} anchor={anchor.strftime('%H:%M')} span={span_ms/60000:.1f}min", flush=True)

    job_ids = []
    t0 = time.time()
    for widx, (win, start, length) in enumerate(plans, 1):
        case, n, cal_title, _ = win
        if case is None:
            print(f"W{widx} 「{cal_title}」no-show 静默窗 {start.strftime('%H:%M')}（不推音频）", flush=True)
            continue
        c = 0
        t = start
        for i in range(1, n + 1):
            ms = chunk_ms(CASES / case, i)
            jid = push_chunk_nopoll(event_id, CASES / case, i, t, ms)
            job_ids.append(jid)
            t += timedelta(milliseconds=ms)
            c += 1
            if not BURST:
                # 串行：等这片转写终态再推下一片——测试任务任意时刻最多占 1 个
                # ASR 槽位，真机分段可随时插队
                while True:
                    st = call("GET", f"/asr-jobs/{jid}")["data"]
                    if st["status"] in ("completed", "failed", "cancelled", "expired"):
                        if st["status"] != "completed":
                            print(f"JOB {jid} 终态={st['status']} err={st.get('errorCode')}", flush=True)
                        break
                    time.sleep(3)
        print(f"W{widx} {case} {c} 片 {start.strftime('%H:%M')}→{t.strftime('%H:%M')} 日程={'「'+cal_title+'」' if cal_title else '无(间隙对话)'}", flush=True)

    # 统一等全部转写终态（走 DB 计数——API 轮询 150 个 job 会打穿限流 429）
    import subprocess
    print(f"uploaded {len(job_ids)} jobs in {(time.time()-t0)/60:.1f}min，等待转写排空…", flush=True)
    t0 = time.time()
    while True:
        out = subprocess.run(
            ["docker", "exec", "memoclipsaas-postgres-1", "psql", "-U", "memoclip", "-d", "memoclip",
             "-t", "-A", "-c",
             f"SELECT count(*) FROM asr_jobs WHERE event_id='{event_id}'"
             " AND status NOT IN ('completed','failed','cancelled','expired')"],
            capture_output=True, text=True)
        pending = int(out.stdout.strip() or -1)
        if pending == 0:
            break
        time.sleep(15)
    print(f"asr drained in {(time.time()-t0)/60:.1f}min", flush=True)
    call("POST", f"/events/{event_id}/recording-finish")
    call("POST", f"/events/{event_id}/extract")
    print("RECORDING FINISHED + EXTRACT", flush=True)
    print(f"EVENT {event_id}", flush=True)


if __name__ == "__main__":
    main()
