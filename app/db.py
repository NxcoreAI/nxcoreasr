"""PostgreSQL 存储：所有租户同表共存（tenant_id 列），RLS 做行级隔离。

隔离规则：应用只用非超级用户 asr_app 连接；每张表 ENABLE + FORCE ROW
LEVEL SECURITY，策略为 tenant_id = current_setting('app.tenant')。
所有访问必须经 tenant_session()（事务内 SET 局部变量）；忘了设置时
current_setting 返回 NULL，一行都匹配不到——fail-closed。
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    Index,
    create_engine,
    event,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
)

from .config import AppConfig


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    version: Mapped[int] = mapped_column(nullable=False, default=1)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    last_active_at: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (
        UniqueConstraint("tenant_id", "external_id"),
        Index("idx_sessions_tenant", "tenant_id", "last_active_at"),
    )


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    seq: Mapped[int | None]  # 四步上传：建单时未定，upload-complete 时才分配
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="queued")
    error: Mapped[str | None]
    audio_path: Mapped[str | None]
    idem_key: Mapped[str | None] = mapped_column(String(128))
    duration_ms: Mapped[int | None]
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    done_at: Mapped[str | None]
    upload_token: Mapped[str | None] = mapped_column(String(64), unique=True)
    upload_expires_at: Mapped[str | None] = mapped_column(String(40))
    hotwords: Mapped[list | None] = mapped_column(JSON)  # 任务级热词（纠偏用）
    __table_args__ = (
        UniqueConstraint("session_id", "idem_key"),
        Index("idx_tasks_session", "session_id", "seq"),
    )


class Person(Base):
    __tablename__ = "persons"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    auto_label: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(64))
    merged_into: Mapped[str | None] = mapped_column(String(64))
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)
    embed_count: Mapped[int] = mapped_column(nullable=False, default=0)
    __table_args__ = (Index("idx_persons_session", "session_id"),)


class Utterance(Base):
    __tablename__ = "utterances"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    seq: Mapped[int] = mapped_column(nullable=False)
    start_ms: Mapped[int | None]
    end_ms: Mapped[int | None]
    text: Mapped[str] = mapped_column(Text, nullable=False)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False)
    __table_args__ = (Index("idx_utt_session", "session_id", "id"),)


class LabelAudit(Base):
    __tablename__ = "label_audit"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    old_value: Mapped[str | None] = mapped_column(String(64))
    new_value: Mapped[str | None] = mapped_column(String(64))
    at: Mapped[str] = mapped_column(String(40), nullable=False)


_RLS_TABLES = ["sessions", "tasks", "persons", "utterances", "label_audit"]


@event.listens_for(Session, "before_flush")
def _stamp_tenant(session, flush_context, instances):
    """新建行自动盖 tenant_id：来源是 tenant_session 绑定的租户上下文。

    业务代码不逐处传 tenant，漏盖会被 RLS 的 WITH CHECK 拒绝（fail-closed）。
    """
    tenant = session.info.get("tenant")
    if not tenant:
        return
    for obj in session.new:
        if isinstance(obj, Base) and obj.tenant_id is None:
            obj.tenant_id = tenant


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def build_engine(cfg: AppConfig):
    return create_engine(cfg.database_url, pool_pre_ping=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for table in _RLS_TABLES:
            conn.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
            conn.execute(text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
            conn.execute(text(f"DROP POLICY IF EXISTS tenant_isolation ON {table}"))
            conn.execute(text(
                f"CREATE POLICY tenant_isolation ON {table}"
                " USING (tenant_id = current_setting('app.tenant', true))"
            ))
        # 存量库补列/放宽约束（create_all 不会改已存在的表）；新库此四句为 no-op
        conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS upload_token varchar(64)"))
        conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS upload_expires_at varchar(40)"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_tasks_upload_token ON tasks(upload_token)"))
        conn.execute(text("ALTER TABLE tasks ALTER COLUMN seq DROP NOT NULL"))
        conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS hotwords jsonb"))


@contextmanager
def tenant_session(engine, tenant: str):
    """带租户上下文的 ORM 会话：事务内绑定 app.tenant，RLS 按行过滤。

    expire_on_commit=False：出块后读已加载属性（如 task.session_id）
    仍是快照值，与原 sqlite3.Row 的用法一致。
    """
    session = Session(engine, expire_on_commit=False)
    session.info["tenant"] = tenant
    try:
        session.execute(text("SELECT set_config('app.tenant', :t, true)"), {"t": tenant})
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def tenant_dir(cfg: AppConfig, tenant: str) -> Path:
    return cfg.data_dir / tenant


def tmp_dir(cfg: AppConfig, tenant: str) -> Path:
    return tenant_dir(cfg, tenant) / "tmp"
