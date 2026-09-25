#!/usr/bin/env python
"""把 memoclip 既有事件的音频原样重灌为新事件（全链路回归：recording-start →
segment complete → asr-jobs → OSS 直传 → 转写 → recording-finish → extract）。

用法：
  .venv/bin/python replay_event_to_memoclip.py <源事件id> [新标题]
  .venv/bin/python replay_event_to_memoclip.py <源事件id> [新标题] probe  # 只推第 1 段

音频从源事件 playback（归一副本优先）presigned URL 下载；时间轴/时长沿用源分段。
"""
import hashlib
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

API = "http://127.0.0.1:4200/api/v1"
USER = "fef6ad8e-7b80-418d-9fe3-cb9660c9d993"
TENANT = "76c2aa13-d19f-4fe8-a899-5f7261340109"
CACHE = Path("/tmp/memoclip-replay-audio")


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


def call(method, path, data=None, headers=None, retries=3):
    h = {"Authorization": f"Bearer {TOK}"}
    if headers:
        h.update(headers)
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        h["Content-Type"] = "application/json"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(API + path, data=body, headers=h, method=method)
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except (ConnectionError, OSError) as e:
            if attempt == retries or "Connection refused" not in str(e):
                raise
            time.sleep(2 * (attempt + 1))


def http_get(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except (ConnectionError, OSError) as e:
            if attempt == 2 or "Connection refused" not in str(e):
                raise
            time.sleep(2 * (attempt + 1))


def fetch_entries(source_event: str):
    data = call("GET", f"/events/{source_event}/playback")["data"]
    entries = sorted(data["entries"], key=lambda x: x["beginsAt"])
    print(f"源事件 {source_event}: {len(entries)} 段, retentionDays={data.get('retentionDays')}")
    return entries


def download(entry) -> bytes:
    CACHE.mkdir(exist_ok=True)
    name = f"{entry['segmentId']}.m4a"
    p = CACHE / name
    if not p.exists():
        p.write_bytes(http_get(entry["url"]))
    return p.read_bytes()


def push_segment(event_id, idx, entry, raw):
    key = f"replay-{entry['segmentId'][:8]}-{uuid.uuid4().hex[:12]}"
    seg = call("POST", f"/events/{event_id}/recording-start", {
        "sourceKind": "phone", "idempotencyKey": key,
        "beganAt": entry["beginsAt"]})["data"]
    seg_id = seg["segment"]["id"] if "segment" in seg else seg["id"]
    call("POST", f"/events/{event_id}/segments/{seg_id}/complete",
         {"endedAt": entry["endsAt"]})
    job = call("POST", "/asr-jobs", {
        "eventId": event_id, "segmentId": seg_id,
        "fileName": f"replay-{idx:02d}.m4a", "mimeType": "audio/mp4",
        "fileSize": len(raw),
        "contentHash": hashlib.sha256(raw).hexdigest(),
        "estimatedDurationMs": entry["durationMs"],
        "idempotencyKey": key, "diarizationEnabled": True})["data"]
    auth = call("POST", f"/asr-jobs/{job['id']}/upload-authorization")["data"]
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
    return st, time.time() - t0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    source = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else None
    probe = len(sys.argv) > 3 and sys.argv[3] == "probe"
    entries = fetch_entries(source)
    if probe:
        entries = entries[:1]
    if not title:
        title = f"重灌实测（源 {source[:8]}）"
    ev = call("POST", "/events", {"title": title})["data"]
    event_id = ev["id"]
    print("EVENT", event_id, flush=True)
    for i, entry in enumerate(entries, 1):
        raw = download(entry)
        st, took = push_segment(event_id, i, entry, raw)
        print(f"seg {i:02d}/{len(entries)}: {st['status']} 耗时{took:.0f}s "
              f"dur={entry['durationMs']}ms size={len(raw)}", flush=True)
        if st["status"] == "failed":
            print("失败详情:", json.dumps(st, ensure_ascii=False)[:400])
            return
    if not probe:
        call("POST", f"/events/{event_id}/recording-finish")
        call("POST", f"/events/{event_id}/extract")
        print("RECORDING FINISHED + EXTRACT", flush=True)
    print("EVENT", event_id, flush=True)


if __name__ == "__main__":
    main()
