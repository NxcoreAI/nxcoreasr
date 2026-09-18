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
        p1 = match_or_create(db, "s1", v, 0.6, 10_000.0, 6_000.0)
        # 很相近的向量 → 同一个人
        v2 = np.zeros(8); v2[0] = 1.0; v2[1] = 0.1
        assert match_or_create(db, "s1", v2, 0.6, 1_000.0, 6_000.0) == p1
        # 正交向量但素材充足 → 新人
        v3 = np.zeros(8); v3[3] = 1.0
        p3 = match_or_create(db, "s1", v3, 0.6, 10_000.0, 6_000.0)
        assert p3 != p1
        # 正交向量但素材不足 → 不新建，挂最像的已有说话人（碎片声纹防线）
        v4 = np.zeros(8); v4[6] = 1.0
        assert match_or_create(db, "s1", v4, 0.6, 2_000.0, 6_000.0) in (p1, p3)
        # 素材不足 + 空库 → 保底新建（首段音频锚点）
        db.add(SessionRecord(id="s2", tenant_id="t", external_id="y", status="active",
                             version=1, created_at=now_iso(), last_active_at=now_iso()))
        v5 = np.zeros(8); v5[0] = 1.0
        assert match_or_create(db, "s2", v5, 0.6, 500.0, 6_000.0).startswith("spk_")

        # 擦边区：最佳相似度落在 [阈值-0.10, 阈值) 且素材充足 → 挂最像的
        # 已有人，不新建也不动质心（防止幽灵声纹随录音数累积）
        from app.db import Person
        db.add(SessionRecord(id="s3", tenant_id="t", external_id="z", status="active",
                             version=1, created_at=now_iso(), last_active_at=now_iso()))
        g1 = np.zeros(8); g1[0] = 1.0
        gp = match_or_create(db, "s3", g1, 0.6, 10_000.0, 6_000.0)
        before = db.get(Person, gp)
        ang = np.deg2rad(55)  # cos55°≈0.574，落在 [0.5, 0.6)
        g2 = np.zeros(8); g2[0] = np.cos(ang); g2[1] = np.sin(ang)
        assert match_or_create(db, "s3", g2, 0.6, 10_000.0, 6_000.0) == gp
        after = db.get(Person, gp)
        assert after.embed_count == before.embed_count == 1
        assert after.centroid == before.centroid
        # 会话内仍只有一个人
        assert len(db.scalars(select(Person).where(Person.session_id == "s3")).all()) == 1
    eng.dispose()


def test_results_filter_by_person(client):
    sid = register(client)
    for _ in range(2):
        wait_done(client, upload(client, sid))
    sp = _speakers(client, sid)
    assert len(sp) == 2
    for s in sp:
        r = client.get(f"/v1/sessions/{sid}/results",
                       params={"person_id": s["person_id"]}, headers=auth()).json()
        # fake engine 每段音频每人一句，两段共两句，且全落在这人名下
        assert len(r["utterances"]) == 2
        assert all(u["person_id"] == s["person_id"] for u in r["utterances"])
        # 两次录音里都有这个人
        assert len({u["task_id"] for u in r["utterances"]}) == 2
    # 不存在的说话人 → 404
    r = client.get(f"/v1/sessions/{sid}/results",
                   params={"person_id": "spk_nope"}, headers=auth())
    assert r.status_code == 404
    # 合并后：按根人过滤应包含被并一方的段落，按被并方过滤则查不到
    p1, p2 = sp[0]["person_id"], sp[1]["person_id"]
    client.post(f"/v1/sessions/{sid}/speakers/merge",
                json={"from_person": p1, "into_person": p2}, headers=auth())
    r = client.get(f"/v1/sessions/{sid}/results",
                   params={"person_id": p2}, headers=auth()).json()
    assert len(r["utterances"]) == 4
    assert all(u["person_id"] == p2 for u in r["utterances"])


def test_speaker_tasks_drill_down(client):
    sid = register(client)
    for _ in range(2):
        wait_done(client, upload(client, sid))
    sp = _speakers(client, sid)
    assert len(sp) == 2
    r = client.get(f"/v1/sessions/{sid}/speakers/{sp[0]['person_id']}/tasks",
                   headers=auth()).json()
    assert r["person_id"] == sp[0]["person_id"]
    # fake engine：每段音频每人一句 → 该人两次录音各 1 句，按 seq 升序
    assert [t["seq"] for t in r["tasks"]] == [1, 2]
    assert all(t["utterance_count"] == 1 and t["spoken_ms"] >= 0 for t in r["tasks"])
    # 不存在的说话人 → 404
    assert client.get(f"/v1/sessions/{sid}/speakers/spk_nope/tasks",
                      headers=auth()).status_code == 404
    # 合并后：根人的录音列表涵盖双方段落
    p1, p2 = sp[0]["person_id"], sp[1]["person_id"]
    client.post(f"/v1/sessions/{sid}/speakers/merge",
                json={"from_person": p1, "into_person": p2}, headers=auth())
    r = client.get(f"/v1/sessions/{sid}/speakers/{p2}/tasks", headers=auth()).json()
    assert all(t["utterance_count"] == 2 for t in r["tasks"])


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
