from tests.conftest import auth, register, upload, KEY_A, KEY_B, sqlite_skip


def test_register_idempotent(client):
    a = register(client, "call-1")
    b = register(client, "call-1")
    assert a == b
    r = client.post("/v1/sessions", json={"external_id": "call-1"}, headers=auth())
    assert r.json()["created"] is False


@sqlite_skip
def test_cross_tenant_isolation(client):
    sid = register(client, "call-1", KEY_A)
    # B 租户看不见 A 的会话，且同名 external_id 互不冲突
    assert client.get(f"/v1/sessions/{sid}/speakers", headers=auth(KEY_B)).status_code == 404
    assert client.delete(f"/v1/sessions/{sid}", headers=auth(KEY_B)).json()["deleted"] is True
    # A 的会话仍然在（B 的"删除"只是幂等空操作）
    assert client.get(f"/v1/sessions/{sid}/speakers", headers=auth(KEY_A)).status_code == 200
    sid_b = register(client, "call-1", KEY_B)
    assert sid_b != sid


def test_no_auth(client):
    assert client.get("/v1/sessions").status_code == 401
    assert client.get("/v1/sessions", headers=auth("wrong")).status_code == 401


def test_health_requires_key(client):
    assert client.get("/v1/health").status_code == 401
    assert client.get("/v1/health", headers=auth("wrong")).status_code == 401
    r = client.get("/v1/health", headers=auth(KEY_A))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["done"] == 0 and body["failed"] == 0


@sqlite_skip
def test_health_only_own_tenant(client):
    """A 的数字不应出现在 B 的 health 里（反之亦然，只统计本租户）。"""
    sid = register(client, "call-1")
    upload(client, sid)
    # 等队列跑完，A 有 done 计数
    import time
    deadline = time.time() + 10
    while time.time() < deadline:
        if client.get("/v1/health", headers=auth(KEY_A)).json()["done"] >= 1:
            break
        time.sleep(0.05)
    assert client.get("/v1/health", headers=auth(KEY_A)).json()["done"] >= 1
    assert client.get("/v1/health", headers=auth(KEY_B)).json()["done"] == 0


def test_config_rejects_placeholder_and_short_keys(tmp_path):
    import pytest
    from app.config import load_config

    def write_config(key_a, key_b="test-key-b", name_a="saas_a"):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(f"""
data_dir: {tmp_path / 'data'}
tenants:
  - name: {name_a}
    api_key: {key_a}
  - name: saas_b
    api_key: {key_b}
""", encoding="utf-8")
        return str(cfg_file)

    with pytest.raises(ValueError, match="占位符"):
        load_config(write_config("CHANGE_ME_KEY_A"))
    with pytest.raises(ValueError, match="至少 8"):
        load_config(write_config("short"))
    with pytest.raises(ValueError, match="不合法"):
        load_config(write_config("test-key-a", name_a="../escape"))
    cfg = load_config(write_config("a-fine-long-key"))
    assert [t.name for t in cfg.tenants] == ["saas_a", "saas_b"]


def test_list_sessions(client):
    register(client, "call-1")
    register(client, "call-2")
    r = client.get("/v1/sessions", headers=auth())
    assert {s["external_id"] for s in r.json()["sessions"]} == {"call-1", "call-2"}
