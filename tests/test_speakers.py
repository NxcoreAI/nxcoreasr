import numpy as np
from sqlalchemy import select

from tests.conftest import TEST_DB_URL, auth, register, upload, wait_done


def _speakers(client, sid):
    return client.get(f"/v1/sessions/{sid}/speakers", headers=auth()).json()["speakers"]


def _version(client, sid):
    return client.get(f"/v1/sessions/{sid}/results", headers=auth()).json()["session_version"]


def test_name_reflects_in_history(client):
    sid = register(client)
    tid = upload(client, sid)
    wait_done(client, tid)
    pid = _speakers(client, sid)[0]["person_id"]
    v0 = _version(client, sid)
    r = client.put(f"/v1/sessions/{sid}/speakers/{pid}/name",
                   json={"name": "张三"}, headers=auth())
    assert r.status_code == 200
    assert _version(client, sid) == v0 + 1
    # 已转完的历史结果里名字联动
    utts = client.get(f"/v1/tasks/{tid}", headers=auth()).json()["utterances"]
    named = [u for u in utts if u["person_id"] == pid]
    assert named and all(u["person_name"] == "张三" and u["named"] for u in named)
    # 清除命名
    client.put(f"/v1/sessions/{sid}/speakers/{pid}/name", json={"name": None}, headers=auth())
    sp = [s for s in _speakers(client, sid) if s["person_id"] == pid][0]
    assert sp["named"] is False


def test_merge_conflict_and_unmerge(client):
    sid = register(client)
    wait_done(client, upload(client, sid))
    sp = _speakers(client, sid)
    p1, p2 = sp[0]["person_id"], sp[1]["person_id"]
    client.put(f"/v1/sessions/{sid}/speakers/{p1}/name", json={"name": "张三"}, headers=auth())
    client.put(f"/v1/sessions/{sid}/speakers/{p2}/name", json={"name": "李四"}, headers=auth())
    # 两侧名字不同 → 拒绝
    r = client.post(f"/v1/sessions/{sid}/speakers/merge",
                    json={"from_person": p1, "into_person": p2}, headers=auth())
    assert r.status_code == 409
    # 清掉一侧后可合并
    client.put(f"/v1/sessions/{sid}/speakers/{p2}/name", json={"name": None}, headers=auth())
    r = client.post(f"/v1/sessions/{sid}/speakers/merge",
                    json={"from_person": p1, "into_person": p2}, headers=auth())
    assert r.status_code == 200
    sp = _speakers(client, sid)
    assert len(sp) == 1
    assert sp[0]["display_name"] == "张三"  # 名字随合并继承
    assert p1 in sp[0]["merged_person_ids"]
    # 历史结果里 p1 的话都归到 p2 名下
    utts = client.get(f"/v1/sessions/{sid}/results", headers=auth()).json()["utterances"]
    assert all(u["person_id"] == p2 for u in utts)
    # 撤销合并
    r = client.post(f"/v1/sessions/{sid}/speakers/{p1}/unmerge", json={}, headers=auth())
    assert r.status_code == 200
    assert len(_speakers(client, sid)) == 2


def test_match_threshold_unit(test_database):
    """直接验证认人归并/新建的数学逻辑。"""
    import numpy as np
    from sqlalchemy import create_engine

    from app.db import Base, SessionRecord, init_db, now_iso, tenant_session
    from app.speakers import match_or_create

    eng = create_engine(TEST_DB_URL)
    Base.metadata.drop_all(eng)
    init_db(eng)
    with tenant_session(eng, "t") as db:
        db.add(SessionRecord(id="s1", tenant_id="t", external_id="x", status="active",
                             version=1, created_at=now_iso(), last_active_at=now_iso()))
        v = np.zeros(8); v[0] = 1.0
        p1 = match_or_create(db, "s1", v, 0.6)
        # 很相近的向量 → 同一个人
        v2 = np.zeros(8); v2[0] = 1.0; v2[1] = 0.1
        assert match_or_create(db, "s1", v2, 0.6) == p1
        # 正交向量 → 新人
        v3 = np.zeros(8); v3[3] = 1.0
        p3 = match_or_create(db, "s1", v3, 0.6)
        assert p3 != p1
    eng.dispose()


def test_audit_trail(client):
    sid = register(client)
    wait_done(client, upload(client, sid))
    pid = _speakers(client, sid)[0]["person_id"]
    client.put(f"/v1/sessions/{sid}/speakers/{pid}/name",
               json={"name": "张三"}, headers=auth())
    # 审计表落了记录（以租户会话直查验证）
    from app.db import LabelAudit, tenant_session
    with tenant_session(client.app.state.db_engine, "saas_a") as db:
        rows = db.scalars(select(LabelAudit)).all()
    assert any(a.action == "name" and a.new_value == "张三" for a in rows)
