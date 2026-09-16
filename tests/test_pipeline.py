from tests.conftest import auth, make_wav, register, upload, wait_done


def test_in_audio_cluster_merge(tmp_path):
    """FunASR 把同一人拆成多个编号 + 短碎片段，应在本音频内合并，不产生幽灵说话人。"""
    from app.engine.base import Segment
    from app.engine.fake_engine import FakeEngine
    from app.service import cluster_local_speakers

    wav_file = tmp_path / "a.wav"
    wav_file.write_bytes(make_wav(seconds=8))  # fake 引擎以前/后半区分音色
    segments = [
        Segment(0, 3000, "长段A", 0),
        Segment(4000, 7000, "长段B", 1),
        Segment(500, 700, "嗯", 2),        # 0.2s 碎片，音色同 spk0
        Segment(5000, 5500, "对", 3),      # 0.5s 碎片，音色同 spk1
        Segment(1000, 3500, "长段A'", 4),  # 同 spk0 音色，被拆成新编号
    ]
    clusters = cluster_local_speakers(FakeEngine(), str(wav_file), segments, 0.6)
    assert len(clusters) == 2
    spk_sets = sorted([set(c["spks"]) for c in clusters], key=lambda s: sorted(s))
    assert spk_sets == [{0, 2, 4}, {1, 3}]


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
