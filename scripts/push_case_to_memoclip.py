#!/usr/bin/env python
"""把四人 data case 经 memoclip App 链路灌进 dev 账号。

用法：
  .venv/bin/python push_to_memoclip.py probe   # 只传第 1 片，打印各步响应结构
  .venv/bin/python push_to_memoclip.py all     # 40 片全量 + recording-finish
"""
import hashlib
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import wave

CASE = Path("/Users/nxcore/Git/nxcoreasr/diarize_cases/four-person-synth-20260921")
API = "http://127.0.0.1:4200/api/v1"
USER = "fef6ad8e-7b80-418d-9fe3-cb9660c9d993"
TENANT = "76c2aa13-d19f-4fe8-a899-5f7261340109"


def jwt_token() -> str:
    import base64, hmac
    env = {}
    for line in open("/Users/nxcore/Git/memoclipsaas/.env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    secret = env["JWT_ACCESS_SECRET"]
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=")
    now = int(time.time())
    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps({"sub": USER, "tid": TENANT, "aud": "memoclip-app",
                           "iat": now, "exp": now + 7200}).encode())
    sig = b64(hmac.new(secret.encode(), head + b"." + body, hashlib.sha256).digest())
    return (head + b"." + body + b"." + sig).decode()


TOK = jwt_token()


def call(method, path, data=None, headers=None):
    h = {"Authorization": f"Bearer {TOK}"}
    if headers:
        h.update(headers)
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(API + path, data=body, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def chunk_ms(i: int) -> int:
    with wave.open(str(CASE / f"cache/{i:02d}.wav"), "rb") as w:
        return int(w.getnframes() / w.getframerate() * 1000)


def push_chunk(event_id, i, probe=False):
    p = CASE / f"chunks/{i:02d}.m4a"
    raw = p.read_bytes()
    key = f"case4p-{i:02d}-{uuid.uuid4().hex[:12]}"
    seg = call("POST", f"/events/{event_id}/recording-start", {
        "sourceKind": "phone", "idempotencyKey": key,
        "beganAt": "2026-09-21T19:00:00.000Z"})["data"]
    seg_id = seg["segment"]["id"] if "segment" in seg else seg["id"]
    call("POST", f"/events/{event_id}/segments/{seg_id}/complete",
         {"endedAt": "2026-09-21T19:00:15.000Z"})
    job = call("POST", "/asr-jobs", {
        "eventId": event_id, "segmentId": seg_id,
        "fileName": f"chunk-{i:02d}.m4a", "mimeType": "audio/mp4",
        "fileSize": len(raw),
        "contentHash": hashlib.sha256(raw).hexdigest(),
        "estimatedDurationMs": chunk_ms(i),
        "idempotencyKey": key, "diarizationEnabled": True})["data"]
    if probe:
        print("job 响应字段:", list(job))
    auth = call("POST", f"/asr-jobs/{job['id']}/upload-authorization")["data"]
    if probe:
        print("授权响应字段:", {k: (v if k != "uploadUrl" else v[:80] + "...") for k, v in auth.items()})
    up = urllib.request.Request(auth["uploadUrl"], data=raw, method="PUT",
                                headers=auth["headers"])
    urllib.request.urlopen(up, timeout=300).read()
    call("POST", f"/asr-jobs/{job['id']}/upload-complete",
         {"objectKey": auth["objectKey"]})
    t0 = time.time()
    while True:
        st = call("GET", f"/asr-jobs/{job['id']}")["data"]
        if st["status"] in ("completed", "failed", "cancelled", "expired"):
            break
        time.sleep(2)
    if probe:
        print("job 终态字段:", {k: st[k] for k in list(st)[:12] if k != "segments"})
    return st, time.time() - t0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    ev = call("POST", "/events", {"title": "四人转写实测（data case 2026-09-21）"})["data"]
    event_id = ev["id"]
    print("EVENT", event_id, flush=True)
    n = 1 if mode == "probe" else 40
    for i in range(1, n + 1):
        st, took = push_chunk(event_id, i, probe=(i == 1))
        print(f"chunk {i:02d}/{n}: {st['status']} 耗时{took:.0f}s", flush=True)
        if st["status"] == "failed":
            print("失败详情:", json.dumps(st, ensure_ascii=False)[:400])
            return
    if mode != "probe":
        call("POST", f"/events/{event_id}/recording-finish")
        # App 录音结束后会主动触发抽取；不调这步事件会一直停在 processing，
        # App 列表看不到（extraction 不自动入队）
        call("POST", f"/events/{event_id}/extract")
        print("RECORDING FINISHED + EXTRACT", flush=True)
        print("EVENT", event_id, flush=True)


if __name__ == "__main__":
    main()
