from tests.conftest import auth, make_wav, register, upload, wait_done


def test_upload_and_transcribe(client):
    sid = register(client)
    tid = upload(client, sid)
    body = wait_done(client, tid)
    assert body["status"] == "done"
    utts = body["utterances"]
    assert len(utts) == 2  # fake engine: 3 秒音频两个说话人各一句
    assert utts[0]["text"]
    assert utts[0]["person_id"] != utts[1]["person_id"]


def test_cross_audio_same_speakers(client):
    """两段音频，fake engine 音色一致 → 会话里始终只有两个人。"""
    sid = register(client)
    t1 = upload(client, sid)
    wait_done(client, t1)
    t2 = upload(client, sid)
    wait_done(client, t2)
    speakers = client.get(f"/v1/sessions/{sid}/speakers", headers=auth()).json()["speakers"]
    assert len(speakers) == 2
    assert all(s["audio_count"] == 2 for s in speakers)


def test_results_cursor(client):
    sid = register(client)
    for _ in range(3):
        wait_done(client, upload(client, sid))
    r = client.get(f"/v1/sessions/{sid}/results", headers=auth()).json()
    assert len(r["utterances"]) == 6
    cursors = [u["cursor"] for u in r["utterances"]]
    assert cursors == sorted(cursors)
    # 增量拉：拿到 next_cursor 后再拉应为空
    r2 = client.get(f"/v1/sessions/{sid}/results",
                    params={"after": r["next_cursor"]}, headers=auth()).json()
    assert r2["utterances"] == []


def test_idempotency_key(client):
    sid = register(client)
    t1 = upload(client, sid, idem="k1")
    t2 = upload(client, sid, idem="k1")
    assert t1 == t2


def test_bad_audio_fails_and_retry_reports(client):
    sid = register(client)
    tid = upload(client, sid, wav=b"not a wav file at all")
    body = wait_done(client, tid)
    assert body["status"] == "failed"
    assert body["error"]
    # 失败任务保留原始音频，可以重试（还是坏文件，会再次失败）
    r = client.post(f"/v1/tasks/{tid}/retry", headers=auth())
    assert r.status_code == 200
    assert wait_done(client, tid)["status"] == "failed"


def test_delete_session_clears_everything(client):
    sid = register(client)
    wait_done(client, upload(client, sid))
    assert client.delete(f"/v1/sessions/{sid}", headers=auth()).json()["deleted"] is True
    assert client.get(f"/v1/sessions/{sid}/results", headers=auth()).status_code == 404
    # 删除后再传报 404（会话已不存在）
    r = client.post(f"/v1/sessions/{sid}/audios",
                    files={"file": ("a.wav", make_wav(), "audio/wav")}, headers=auth())
    assert r.status_code == 404
    # external_id 释放，可重新注册为全新会话
    sid2 = register(client)
    assert sid2 != sid
