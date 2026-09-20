"""配置加载：config.yaml + 少量环境变量。"""
import hmac
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TENANT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class Tenant:
    name: str
    api_key: str


@dataclass
class OssConfig:
    """阿里云 OSS：四项核心齐备才启用，否则本地盘回退。"""

    endpoint: str
    bucket: str
    access_key_id: str
    access_key_secret: str
    upload_ttl_seconds: int = 600


DEFAULT_DATABASE_URL = "postgresql+psycopg://asr_app:asr_dev_password@127.0.0.1:25432/asr"


@dataclass
class AppConfig:
    data_dir: Path
    engine: str = "llamacpp"
    llamacpp_bin: str = ""
    llamacpp_dir: str = ""
    llamacpp_llm: str = ""
    model_dir: Path = Path("./models")
    max_workers: int = 2
    llm_quantize: bool = False
    speaker_threshold: float = 0.45
    speaker_window_threshold: float = 0.65
    speaker_min_enroll_ms: int = 6000
    max_upload_mb: int = 50
    database_url: str = DEFAULT_DATABASE_URL
    oss: OssConfig | None = None
    tenants: list[Tenant] = field(default_factory=list)

    def tenant_by_key(self, api_key: str) -> Tenant | None:
        for t in self.tenants:
            if hmac.compare_digest(t.api_key, api_key):
                return t
        return None


def _validate_tenant(t: Tenant) -> None:
    if not TENANT_NAME_RE.match(t.name):
        raise ValueError(
            f"租户名 {t.name!r} 不合法（只允许字母、数字、下划线、连字符），"
            "它是数据目录名，含其他字符会被拒绝"
        )
    if not t.api_key or len(t.api_key) < 8:
        raise ValueError(f"租户 {t.name} 的 api_key 至少 8 个字符")
    if t.api_key.startswith("CHANGE_ME"):
        raise ValueError(
            f"租户 {t.name} 的 api_key 还是占位符，请生成随机长串"
            "（如 openssl rand -hex 32）后写入 config.yaml"
        )


def load_config(path: str | None = None) -> AppConfig:
    path = path or os.environ.get("ASR_CONFIG", "config.yaml")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    tenants = [Tenant(name=t["name"], api_key=str(t["api_key"])) for t in raw.get("tenants", [])]
    if not tenants:
        raise ValueError("config.yaml 中未配置任何租户")
    names = [t.name for t in tenants]
    if len(names) != len(set(names)):
        raise ValueError("租户名重复")
    keys = [t.api_key for t in tenants]
    if len(keys) != len(set(keys)):
        raise ValueError("租户 api_key 重复")
    for t in tenants:
        _validate_tenant(t)
    model_dir = Path(raw.get("model_dir", "./models")).resolve()
    # 引擎资产默认放在 model_dir/llamacpp/ 下（scripts/setup_llamacpp.sh 的产出）
    lc = model_dir / "llamacpp"
    bin_default = str(lc / "bin" / "llama-funasr-cli")
    dir_default = str(lc / "gguf")
    import platform

    # 老至强 x86 上 q4 明显快于 q8；Apple 芯片两者同速、q8 文本更准
    llm_default = ("qwen3-0.6b-q4km.gguf"
                   if platform.machine() in ("x86_64", "AMD64")
                   else "qwen3-0.6b-q8_0.gguf")
    oss_raw = raw.get("oss") or {}
    oss = OssConfig(
        endpoint=os.environ.get("ASR_OSS_ENDPOINT") or str(oss_raw.get("endpoint", "")),
        bucket=os.environ.get("ASR_OSS_BUCKET") or str(oss_raw.get("bucket", "")),
        access_key_id=os.environ.get("ASR_OSS_ACCESS_KEY_ID") or str(oss_raw.get("access_key_id", "")),
        access_key_secret=(os.environ.get("ASR_OSS_ACCESS_KEY_SECRET")
                           or str(oss_raw.get("access_key_secret", ""))),
        upload_ttl_seconds=int(os.environ.get("ASR_OSS_UPLOAD_TTL_SECONDS")
                               or oss_raw.get("upload_ttl_seconds", 600)),
    )
    if not all((oss.endpoint, oss.bucket, oss.access_key_id, oss.access_key_secret)):
        oss = None
    return AppConfig(
        data_dir=Path(raw.get("data_dir", "./data")).resolve(),
        model_dir=model_dir,
        engine=str(raw.get("engine", "llamacpp")),
        llamacpp_bin=str(raw.get("llamacpp_bin", bin_default)),
        llamacpp_dir=str(raw.get("llamacpp_dir", dir_default)),
        llamacpp_llm=str(raw.get("llamacpp_llm", llm_default)),
        max_workers=int(raw.get("max_workers", 2)),
        llm_quantize=bool(raw.get("llm_quantize", False)),
        speaker_threshold=float(raw.get("speaker_threshold", 0.45)),
        speaker_window_threshold=float(raw.get("speaker_window_threshold", 0.65)),
        speaker_min_enroll_ms=int(raw.get("speaker_min_enroll_ms", 6000)),
        max_upload_mb=int(raw.get("max_upload_mb", 50)),
        database_url=os.environ.get("ASR_DATABASE_URL")
        or str(raw.get("database_url", DEFAULT_DATABASE_URL)),
        oss=oss,
        tenants=tenants,
    )
