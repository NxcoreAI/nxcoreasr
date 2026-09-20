"""对象存储：OSS 真后端 + 本地盘回退。

未配 oss 段走本地盘（授权地址指向本服务，字节落 data/{tenant}/tmp）；
配了走阿里云 OSS，对象落 nxcore/ 前缀，客户端字节直达对象存储。
桶内 audio/ 是长期备份、asr-staging/ 是 SaaS 转写暂存，nxcore/ 只放
待转写音频，转完即删；孤儿对象靠 nxcore/ 前缀级生命周期兜底清理
（严禁桶级过期规则，会连长期备份一起删）。
"""
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig


class ObjectNotFound(Exception):
    pass


@dataclass
class UploadTarget:
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    direct: bool = False  # True=字节直达对象存储，不经本服务


class LocalObjectStorage:
    """本地盘模式：授权地址指向本服务 PUT 接口，key 即本地绝对路径。"""

    direct = False
    upload_ttl = 600

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def upload_target(self, key: str, token: str, ttl: int, base_url: str) -> UploadTarget:
        return UploadTarget(url=f"{base_url}/v1/uploads/{token}")

    def head_size(self, key: str) -> int:
        p = Path(key)
        if not p.is_file():
            raise ObjectNotFound(key)
        return p.stat().st_size

    def exists(self, key: str) -> bool:
        return Path(key).is_file()

    def materialize(self, key: str) -> str:
        return key

    def delete(self, key: str) -> None:
        Path(key).unlink(missing_ok=True)


class AliyunOssStorage:
    """OSS 模式：预签名 PUT 直传，转写前下载到本地临时目录。"""

    direct = True

    def __init__(self, cfg: AppConfig):
        import oss2  # 延迟导入：local 模式不装 oss2 也能跑

        self._oss2 = oss2
        self._bucket = oss2.Bucket(
            oss2.Auth(cfg.oss.access_key_id, cfg.oss.access_key_secret),
            cfg.oss.endpoint, cfg.oss.bucket)
        self._download_dir = cfg.data_dir / "oss-downloads"
        self.upload_ttl = cfg.oss.upload_ttl_seconds

    def upload_target(self, key: str, token: str, ttl: int, base_url: str) -> UploadTarget:
        headers = {"Content-Type": "application/octet-stream"}
        url = self._bucket.sign_url("PUT", key, ttl, slash_safe=True, headers=headers)
        return UploadTarget(url=url, headers=headers, direct=True)

    def head_size(self, key: str) -> int:
        try:
            return self._bucket.head_object(key).content_length
        except (self._oss2.exceptions.NoSuchKey, self._oss2.exceptions.NotFound):
            raise ObjectNotFound(key)

    def exists(self, key: str) -> bool:
        try:
            self.head_size(key)
            return True
        except ObjectNotFound:
            return False

    def materialize(self, key: str) -> str:
        self._download_dir.mkdir(parents=True, exist_ok=True)
        dest = self._download_dir / key.replace("/", "_")
        self._bucket.get_object_to_file(key, dest)
        return str(dest)

    def delete(self, key: str) -> None:
        self._bucket.delete_object(key)  # 对不存在对象同样成功，天然幂等
