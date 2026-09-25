#!/usr/bin/env python
"""多个 data case 拼接成单场「不中断长录音」全链路测试。

一个事件顺序推完所有 case 的全部 chunk（时间轴零间隙、连续推进），
最后 recording-finish + extract——测长事件的转写累积、说话人轨、
清扫器不误伤，以及 agent 自动拆分建议（多议题一场的正样本）。

用法：
  .venv/bin/python push_cases_combined.py <title> <case_dir1> <case_dir2> ...
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from push_case_to_memoclip import call, push_chunk


def main():
    args = [a for a in sys.argv[1:]]
    if len(args) < 2:
        print(__doc__)
        sys.exit(2)
    title, case_dirs = args[0], [Path(a).resolve() for a in args[1:]]
    for case in case_dirs:
        if not (case / "chunks").is_dir():
            print(f"case 目录不存在或缺 chunks/: {case}")
            sys.exit(2)

    ev = call("POST", "/events", {"title": title})["data"]
    event_id = ev["id"]
    counts = [len(list((c / "chunks").glob("*.m4a"))) for c in case_dirs]
    total = sum(counts)
    print(f"EVENT {event_id} cases={len(case_dirs)} chunks={total}", flush=True)

    cursor = datetime.now(timezone.utc)
    n = 0
    t0 = time.time()
    for case, count in zip(case_dirs, counts):
        for i in range(1, count + 1):
            n += 1
            st, took, dur_ms = push_chunk(case, case.name, event_id, i, cursor)
            cursor += timedelta(milliseconds=dur_ms)
            span = int((cursor - datetime.now(timezone.utc)).total_seconds())
            print(f"[{case.name[-16:-9]}] {i:02d}/{count} ({n}/{total}): {st['status']} "
                  f"耗时{took:.0f}s 时间轴领先{span}s", flush=True)
            if st["status"] == "failed":
                print("失败详情:", json.dumps(st, ensure_ascii=False)[:400])
                return
    call("POST", f"/events/{event_id}/recording-finish")
    call("POST", f"/events/{event_id}/extract")
    print(f"RECORDING FINISHED + EXTRACT 总墙钟{(time.time()-t0)/60:.0f}min", flush=True)
    print(f"EVENT {event_id}", flush=True)


if __name__ == "__main__":
    main()
