"""AliyunOssStorage 单测：stub 掉 oss2，不碰真实 OSS。"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.storage import AliyunOssStorage, ObjectNotFound


@pytest.fixture()
def fake_oss2(monkeypatch):
    store: dict[str, bytes] = {}
    signed: list[dict] = []

    exc = types.ModuleType("oss2.exceptions")

    class NoSuchKey(Exception):
        pass

    class NotFound(Exception):
        pass

    exc.NoSuchKey = NoSuchKey
    exc.NotFound = NotFound

    oss2 = types.ModuleType("oss2")

    class Auth:
        def __init__(self, ak, sk):
            pass

    class Bucket:
        def __init__(self, auth, endpoint, bucket):
            pass

        def sign_url(self, method, key, expires, slash_safe=False, headers=None):
            signed.append({"method": method, "key": key, "expires": expires,
                           "headers": headers})
            return f"https://oss.example.com/{key}?Expires={expires}&Signature=x"

        def head_object(self, key):
            if key not in store:
                raise NoSuchKey(key)
            return SimpleNamespace(content_length=len(store[key]))

        def get_object_to_file(self, key, dest):
            if key not in store:
                raise NoSuchKey(key)
            Path(dest).write_bytes(store[key])

        def delete_object(self, key):
            store.pop(key, None)

    oss2.Auth = Auth
    oss2.Bucket = Bucket
    oss2.exceptions = exc
    monkeypatch.setitem(sys.modules, "oss2", oss2)
    monkeypatch.setitem(sys.modules, "oss2.exceptions", exc)
    return SimpleNamespace(store=store, signed=signed)


def _storage(tmp_path):
    from app.config import AppConfig, OssConfig
    cfg = AppConfig(data_dir=tmp_path,
                    oss=OssConfig(endpoint="https://oss.example.com", bucket="b",
                                  access_key_id="ak", access_key_secret="sk"))
    return AliyunOssStorage(cfg)


def test_upload_target_signs_put(fake_oss2, tmp_path):
    s = _storage(tmp_path)
    t = s.upload_target("nxcore/t/j/tok", "tok", 600, "http://127.0.0.1:8300")
    assert t.direct is True
    assert t.headers == {"Content-Type": "application/octet-stream"}
    assert fake_oss2.signed[-1] == {"method": "PUT", "key": "nxcore/t/j/tok",
                                    "expires": 600,
                                    "headers": {"Content-Type": "application/octet-stream"}}
    assert "nxcore/t/j/tok" in t.url


def test_head_size_and_exists(fake_oss2, tmp_path):
    s = _storage(tmp_path)
    with pytest.raises(ObjectNotFound):
        s.head_size("k")
    fake_oss2.store["k"] = b"abc"
    assert s.head_size("k") == 3
    assert s.exists("k") is True
    assert s.exists("nope") is False


def test_materialize_flattens_key(fake_oss2, tmp_path):
    s = _storage(tmp_path)
    fake_oss2.store["nxcore/t/j/x"] = b"hello"
    p = s.materialize("nxcore/t/j/x")
    assert Path(p).read_bytes() == b"hello"
    assert "nxcore_t_j_x" in p


def test_delete_idempotent(fake_oss2, tmp_path):
    s = _storage(tmp_path)
    fake_oss2.store["k"] = b"x"
    s.delete("k")
    s.delete("k")
    assert "k" not in fake_oss2.store
