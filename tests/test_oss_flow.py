"""OSS 直传模式的主流程测试：monkeypatch 掉 AliyunOssStorage，不碰真实 OSS。"""
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient

from tests.conftest import KEY_A, KEY_B, TEST_DB_URL, auth, create_job, make_wav, wait_done
from app.storage import ObjectNotFound, UploadTarget


class FakeOssStorage:
    """dict 充当对象存储；materialize 把字节落成真文件供 ffmpeg/fake 引擎走通。"""

    direct = True
    upload_ttl = 600
    blobs: dict[str, bytes] = {}

    def __init__(self, cfg):
        self.cfg = cfg

    def upload_target(self, key, token, ttl, base_url):
        return UploadTarget(url=f"https://oss-fake/{key}",
                            headers={"Content-Type": "application/octet-stream"},
                            direct=True)

    def head_size(self, key):
        if key not in self.blobs:
            raise ObjectNotFound(key)
        return len(self.blobs[key])

    def exists(self, key):
        return key in self.blobs

    def materialize(self, key):
        d = self.cfg.data_dir / "fake-oss"
        d.mkdir(parents=True, exist_ok=True)
        p = d / key.replace("/", "_")
        p.write_bytes(self.blobs[key])
        return str(p)

    def delete(self, key):
        self.blobs.pop(key, None)


@pytest.fixture()
def client_oss(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.AliyunOssStorage", FakeOssStorage)
    FakeOssStorage.blobs.clear()
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
max_upload_mb: 5
database_url: {TEST_DB_URL}
oss:
  endpoint: https://oss.example.com
  bucket: fake-bucket
  access_key_id: ak
  access_key_secret: sk
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
    FakeOssStorage.blobs.clear()


def _authorize_key(client, job_id) -> str:
    r = client.post(f"/v1/asr-jobs/{job_id}/upload-authorization", headers=auth())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "PUT"
    assert body["headers"]["Content-Type"] == "application/octet-stream"
    return urlparse(body["upload_url"]).path.lstrip("/")


def test_oss_direct_flow_completes_and_deletes_object(client_oss):
    job = create_job(client_oss, "oss-1")
    key = _authorize_key(client_oss, job["job_id"])
    assert key.startswith("nxcore/saas_a/")
    FakeOssStorage.blobs[key] = make_wav()  # 客户端字节直达"OSS"

    c = client_oss.post(f"/v1/asr-jobs/{job['job_id']}/upload-complete", headers=auth())
    assert c.status_code == 200 and c.json()["queued"] is True

    done = wait_done(client_oss, job["job_id"])
    assert done["status"] == "done" and done["utterances"]
    assert key not in FakeOssStorage.blobs  # 转写成功即删对象


def test_oss_complete_without_object(client_oss):
    job = create_job(client_oss, "oss-miss-1")
    _authorize_key(client_oss, job["job_id"])
    r = client_oss.post(f"/v1/asr-jobs/{job['job_id']}/upload-complete", headers=auth())
    assert r.status_code == 409


def test_oss_complete_empty_object(client_oss):
    job = create_job(client_oss, "oss-empty-1")
    key = _authorize_key(client_oss, job["job_id"])
    FakeOssStorage.blobs[key] = b""
    r = client_oss.post(f"/v1/asr-jobs/{job['job_id']}/upload-complete", headers=auth())
    assert r.status_code == 400


def test_oss_complete_oversize_deletes_object(client_oss):
    job = create_job(client_oss, "oss-big-1")
    key = _authorize_key(client_oss, job["job_id"])
    FakeOssStorage.blobs[key] = b"x" * (5 * 1024 * 1024 + 1)  # max_upload_mb: 5
    r = client_oss.post(f"/v1/asr-jobs/{job['job_id']}/upload-complete", headers=auth())
    assert r.status_code == 413
    assert key not in FakeOssStorage.blobs  # 超限对象当场清掉


def test_oss_reauthorize_orphans_old_key(client_oss):
    job = create_job(client_oss, "oss-reauth-1")
    k1 = _authorize_key(client_oss, job["job_id"])
    k2 = _authorize_key(client_oss, job["job_id"])
    assert k1 != k2
    FakeOssStorage.blobs[k2] = make_wav()
    r = client_oss.post(f"/v1/asr-jobs/{job['job_id']}/upload-complete", headers=auth())
    assert r.status_code == 200  # 只认最新授权的 key
