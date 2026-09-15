from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from tests.conftest import (
    KEY_A,
    KEY_B,
    auth,
    create_job,
    make_wav,
    register,
    upload,
    wait_done,
)


def _authorize(client, job, key=KEY_A) -> dict:
    r = client.post(f"/v1/asr-jobs/{job}/upload-authorization", headers=auth(key))
    assert r.status_code == 200, r.text
    return r.json()


def _put(client, url, wav=None, key=KEY_A):
    return client.put(urlparse(url).path, content=wav if wav is not None else make_wav(),
                      headers={**auth(key), "Content-Type": "application/octet-stream"})


def test_four_step_flow(client):
    r = client.post("/v1/asr-jobs", json={"external_id": "meeting-1"}, headers=auth())
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] == "pending_upload" and job["created"] is True

    a = _authorize(client, job["job_id"])
    assert a["method"] == "PUT" and "/v1/uploads/" in a["upload_url"]

    p = _put(client, a["upload_url"])
    assert p.status_code == 200 and p.json()["status"] == "uploaded"

    c = client.post(f"/v1/asr-jobs/{job['job_id']}/upload-complete", headers=auth())
    assert c.status_code == 200 and c.json()["queued"] is True

    done = wait_done(client, job["job_id"])
    assert done["status"] == "done" and done["utterances"]
    # 轮询别名与 /tasks 完全同构
    alias = client.get(f"/v1/asr-jobs/{job['job_id']}", headers=auth()).json()
    assert alias["status"] == "done" and alias["utterances"]

    # 同 external_id 再建一个 job → 落同一会话（会话相同）
    second = client.post("/v1/asr-jobs", json={"external_id": "meeting-1"},
                         headers=auth()).json()
    assert second["session_id"] == job["session_id"]


def test_complete_before_upload_rejected(client):
    job = create_job(client, "early-1")["job_id"]
    r = client.post(f"/v1/asr-jobs/{job}/upload-complete", headers=auth())
    assert r.status_code == 409


def test_complete_idempotent_after_done(client):
    sid = register(client, "idem-1")
    job = upload(client, sid)
    wait_done(client, job)
    r = client.post(f"/v1/asr-jobs/{job}/upload-complete", headers=auth())
    assert r.status_code == 200
    assert r.json()["queued"] is False and r.json()["status"] == "done"


def test_token_invisible_to_other_tenant(client):
    job = create_job(client, "iso-1", key=KEY_A)["job_id"]
    a = _authorize(client, job, key=KEY_A)
    assert _put(client, a["upload_url"], key=KEY_B).status_code == 404


def test_reauthorize_invalidates_old_token(client):
    job = create_job(client, "reauth-1")["job_id"]
    a1 = _authorize(client, job)
    a2 = _authorize(client, job)
    assert a1["upload_url"] != a2["upload_url"]
    assert _put(client, a1["upload_url"]).status_code == 404
    assert _put(client, a2["upload_url"]).status_code == 200


def test_expired_token_rejected(client):
    job = create_job(client, "exp-1")["job_id"]
    a = _authorize(client, job)
    from app.db import Task, tenant_session
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(timespec="microseconds")
    with tenant_session(client.app.state.db_engine, "saas_a") as db:
        db.get(Task, job).upload_expires_at = past
    assert _put(client, a["upload_url"]).status_code == 410


def test_empty_put_rejected(client):
    job = create_job(client, "empty-1")["job_id"]
    a = _authorize(client, job)
    assert _put(client, a["upload_url"], wav=b"").status_code == 400


def test_job_idempotency_key(client):
    sid = register(client, "idemkey-1")
    j1 = client.post("/v1/asr-jobs", json={"external_id": "idemkey-1"},
                     headers={**auth(), "Idempotency-Key": "k1"}).json()
    j2 = client.post("/v1/asr-jobs", json={"external_id": "idemkey-1"},
                     headers={**auth(), "Idempotency-Key": "k1"}).json()
    assert j1["job_id"] == j2["job_id"] and j2["duplicated"] is True
    # 幂等返回的 job 仍然可以走完后续步骤
    a = _authorize(client, j2["job_id"])
    assert _put(client, a["upload_url"]).status_code == 200


def test_legacy_upload_endpoint_removed(client):
    sid = register(client, "legacy-1")
    r = client.post(f"/v1/sessions/{sid}/audios",
                    files={"file": ("a.wav", make_wav(), "audio/wav")}, headers=auth())
    assert r.status_code == 404


def test_job_requires_key(client):
    assert client.post("/v1/asr-jobs", json={"external_id": "x"}).status_code == 401
    job = create_job(client, "auth-1")["job_id"]
    assert client.post(f"/v1/asr-jobs/{job}/upload-authorization").status_code == 401
    a = _authorize(client, job)
    assert client.put(urlparse(a["upload_url"]).path, content=make_wav()).status_code == 401
    assert client.post(f"/v1/asr-jobs/{job}/upload-complete").status_code == 401


def test_list_tasks(client):
    sid = register(client, "list-1")
    done = upload(client, sid)
    wait_done(client, done)
    pending = create_job(client, "list-1")["job_id"]  # 只建单不传，停在 pending_upload
    r = client.get(f"/v1/sessions/{sid}/tasks", headers=auth())
    assert r.status_code == 200, r.text
    tasks = {t["task_id"]: t for t in r.json()["tasks"]}
    assert set(tasks) == {done, pending}
    assert tasks[done]["status"] == "done" and tasks[done]["seq"] == 1
    assert tasks[done]["duration_ms"] is not None
    assert tasks[pending]["status"] == "pending_upload" and tasks[pending]["seq"] is None
    # 别的租户看不见
    assert client.get(f"/v1/sessions/{sid}/tasks", headers=auth(KEY_B)).status_code == 404
