import io
import os
import struct
import wave

import pytest
from fastapi.testclient import TestClient

os.environ["ASR_FAKE_ENGINE"] = "1"

KEY_A = "test-key-a"
KEY_B = "test-key-b"

TEST_DB_URL = "postgresql+psycopg://asr_app:asr_dev_password@127.0.0.1:25432/asr_test"
ADMIN_URL = "postgresql+psycopg://asr_admin:asr_admin_dev_only@127.0.0.1:25432/postgres"


@pytest.fixture(scope="session", autouse=True)
def test_database():
    """session 级：以超管建测试库（owner 为 asr_app），复用则跳过建库。"""
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool

    admin = create_engine(ADMIN_URL, poolclass=NullPool,
                          isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        exists = c.execute(
            text("SELECT 1 FROM pg_database WHERE datname = 'asr_test'")
        ).scalar()
        if not exists:
            c.execute(text("CREATE DATABASE asr_test OWNER asr_app"))
    admin.dispose()
    yield


@pytest.fixture()
def client(tmp_path):
    # 每个测试重置 schema（等价于原来每测试一个新 SQLite 文件）
    from sqlalchemy import create_engine

    from app.db import Base, init_db

    eng = create_engine(TEST_DB_URL)
    Base.metadata.drop_all(eng)
    init_db(eng)
    eng.dispose()

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"""
data_dir: {tmp_path / 'data'}
max_workers: 2
speaker_threshold: 0.6
max_upload_mb: 5
database_url: {TEST_DB_URL}
tenants:
  - name: saas_a
    api_key: {KEY_A}
  - name: saas_b
    api_key: {KEY_B}
""", encoding="utf-8")
    from app.main import create_app
    app = create_app(str(cfg_file))
    with TestClient(app) as c:
        yield c


def auth(key=KEY_A):
    return {"Authorization": f"Bearer {key}"}


def make_wav(seconds: float = 16.0, rate: int = 16000) -> bytes:
    """静音 PCM16 wav（fake engine 只看时长）；默认 16s 使每个假说话人
    素材 8s，过 speaker_min_enroll_ms 门槛正常建卡。"""
    buf = io.BytesIO()
    n = int(seconds * rate)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return buf.getvalue()


def register(client, external_id="call-1", key=KEY_A) -> str:
    r = client.post("/v1/sessions", json={"external_id": external_id}, headers=auth(key))
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _ext_of(client, sid, key=KEY_A) -> str:
    r = client.get("/v1/sessions", headers=auth(key))
    for s in r.json()["sessions"]:
        if s["session_id"] == sid:
            return s["external_id"]
    raise AssertionError(f"会话不存在: {sid}")


def create_job(client, external_id, key=KEY_A, idem=None) -> dict:
    headers = auth(key)
    if idem:
        headers["Idempotency-Key"] = idem
    r = client.post("/v1/asr-jobs", json={"external_id": external_id}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def upload(client, sid, key=KEY_A, wav=None, idem=None) -> str:
    """四步上传：建 job → 授权 → PUT 裸字节 → 确认入队；返回 job_id。"""
    from urllib.parse import urlparse
    job = create_job(client, _ext_of(client, sid, key), key, idem)
    if job["status"] not in ("pending_upload", "uploaded"):
        return job["job_id"]  # 幂等命中已入队/已完成的 job，直接轮询即可
    a = client.post(f"/v1/asr-jobs/{job['job_id']}/upload-authorization", headers=auth(key))
    assert a.status_code == 200, a.text
    p = client.put(urlparse(a.json()["upload_url"]).path,
                   content=wav or make_wav(),
                   headers={**auth(key), "Content-Type": "application/octet-stream"})
    assert p.status_code == 200, p.text
    c = client.post(f"/v1/asr-jobs/{job['job_id']}/upload-complete", headers=auth(key))
    assert c.status_code == 200, c.text
    return job["job_id"]


def wait_done(client, tid, key=KEY_A, timeout=10.0) -> dict:
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/tasks/{tid}", headers=auth(key))
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("任务超时未完成")
