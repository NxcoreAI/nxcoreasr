"""FastAPI 应用与全部路由。

隔离规则：租户只从 Bearer key 推导（auth.py）；所有租户数据同表共存，
靠 PG 行级安全（RLS）按 tenant_id 隔离——handler 一律通过
tenant_session 访问数据，跨租户资源天然查不到，一律 404，不暴露
"存在但无权"。
"""
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from . import speakers as spk
from .auth import require_tenant
from .config import load_config
from .db import (
    Person,
    SessionRecord,
    Task,
    Utterance,
    build_engine,
    init_db,
    now_iso,
    tenant_session,
    tmp_dir,
)
from .models import CreateJobReq, MergeReq, NameReq, RegisterSessionReq
from .queueing import Scheduler
from .speakers import SpeakerError

log = logging.getLogger(__name__)

UPLOAD_TOKEN_TTL = 600  # 上传授权 URL 有效期（秒）


def _new_id(prefix: str) -> str:
    return prefix + "_" + secrets.token_urlsafe(16)


def _build_engine_(cfg):
    if os.environ.get("ASR_FAKE_ENGINE") == "1":
        from .engine.fake_engine import FakeEngine
        return FakeEngine()
    if cfg.engine == "llamacpp":
        if not (cfg.llamacpp_bin and cfg.llamacpp_dir):
            raise ValueError("engine=llamacpp 需要同时配置 llamacpp_bin 和 llamacpp_dir")
        from .engine.llamacpp_engine import LlamaCppEngine
        return LlamaCppEngine(cfg.model_dir, cfg.llamacpp_bin, cfg.llamacpp_dir,
                              cfg.llamacpp_llm, cfg.speaker_window_threshold)
    from .engine.funasr_engine import FunasrEngine
    return FunasrEngine(cfg.model_dir, cfg.speaker_window_threshold, cfg.llm_quantize)


def create_app(config_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = load_config(config_path)
        for t in cfg.tenants:
            tmp_dir(cfg, t.name).mkdir(parents=True, exist_ok=True)
        app.state.config = cfg
        app.state.db_engine = build_engine(cfg)
        init_db(app.state.db_engine)
        app.state.engine = _build_engine_(cfg)
        app.state.scheduler = Scheduler(cfg, app.state.engine, app.state.db_engine)
        app.state.scheduler.start()
        yield
        app.state.scheduler.stop()
        app.state.db_engine.dispose()

    app = FastAPI(title="nxcore-asr", lifespan=lifespan)

    def _get_session(db_engine, tenant: str, sid: str) -> SessionRecord:
        with tenant_session(db_engine, tenant) as db:
            rec = db.get(SessionRecord, sid)
        if rec is None:
            raise HTTPException(404, "会话不存在")
        return rec

    # ---- 会话 ----

    def _get_or_create_session(db, tenant: str, external_id: str) -> tuple[SessionRecord, bool]:
        """按 external_id 幂等 get-or-create；返回 (记录, 是否新建)。

        并发撞车用 SAVEPOINT 兜底：只回滚嵌套事务，保住外层——
        外层事务里带着 RLS 的租户上下文，整个 rollback 会把它一起丢掉。
        """
        row = db.scalars(select(SessionRecord)
                         .where(SessionRecord.external_id == external_id)).first()
        if row:
            if row.status != "active":
                raise HTTPException(409, "同名会话正在删除中，请稍后重试")
            return row, False
        sid = _new_id("ses")
        ts = now_iso()
        rec = SessionRecord(id=sid, tenant_id=tenant, external_id=external_id,
                            status="active", version=1, created_at=ts, last_active_at=ts)
        try:
            with db.begin_nested():
                db.add(rec)
        except IntegrityError:
            row = db.scalars(select(SessionRecord)
                             .where(SessionRecord.external_id == external_id)).first()
            return row, False
        return rec, True

    @app.post("/v1/sessions")
    def register_session(body: RegisterSessionReq, request: Request,
                         tenant: str = Depends(require_tenant)):
        with tenant_session(request.app.state.db_engine, tenant) as db:
            rec, created = _get_or_create_session(db, tenant, body.external_id)
        return {"session_id": rec.id, "external_id": rec.external_id,
                "created": created, "version": rec.version}

    @app.get("/v1/sessions")
    def list_sessions(request: Request, tenant: str = Depends(require_tenant)):
        with tenant_session(request.app.state.db_engine, tenant) as db:
            rows = db.execute(
                select(
                    SessionRecord,
                    select(func.count()).select_from(Task)
                        .where(Task.session_id == SessionRecord.id).correlate(SessionRecord)
                        .scalar_subquery().label("audio_count"),
                    select(func.count()).select_from(Task)
                        .where(Task.session_id == SessionRecord.id, Task.status == "done")
                        .correlate(SessionRecord).scalar_subquery().label("done_count"),
                    select(func.count()).select_from(Task)
                        .where(Task.session_id == SessionRecord.id, Task.status == "failed")
                        .correlate(SessionRecord).scalar_subquery().label("failed_count"),
                    select(func.count()).select_from(Person)
                        .where(Person.session_id == SessionRecord.id,
                               Person.merged_into.is_(None))
                        .correlate(SessionRecord).scalar_subquery().label("speaker_count"),
                ).order_by(SessionRecord.last_active_at.desc())
            ).all()
        return {"sessions": [{
            "session_id": r[0].id, "external_id": r[0].external_id, "status": r[0].status,
            "version": r[0].version, "audio_count": r[1],
            "done_count": r[2], "failed_count": r[3],
            "speaker_count": r[4],
            "created_at": r[0].created_at, "last_active_at": r[0].last_active_at,
        } for r in rows]}

    @app.delete("/v1/sessions/{sid}")
    def delete_session(sid: str, request: Request, tenant: str = Depends(require_tenant)):
        cfg = request.app.state.config
        with tenant_session(request.app.state.db_engine, tenant) as db:
            rec = db.get(SessionRecord, sid)
            if rec is None:
                return {"deleted": True}  # 幂等
            rec.status = "deleting"
        finished = request.app.state.scheduler.cancel_session(tenant, sid)
        if not finished:
            raise HTTPException(503, "会话有任务长时间未收尾，请稍后重试删除")
        with tenant_session(request.app.state.db_engine, tenant) as db:
            paths = [r[0] for r in db.execute(
                select(Task.audio_path).where(Task.session_id == sid,
                                              Task.audio_path.is_not(None))).all()]
            for model in (Utterance, Person, Task):
                db.execute(delete(model).where(model.session_id == sid))
            db.execute(delete(SessionRecord).where(SessionRecord.id == sid))
        for p in paths:
            Path(p).unlink(missing_ok=True)
        log.info("会话已删除 tenant=%s session=%s", tenant, sid)
        return {"deleted": True}

    # ---- 音频与任务 ----

    # ---- 转写任务（四步上传：建单 → 授权 → PUT 直传 → 确认） ----

    def _get_job(db, jid: str) -> Task:
        task = db.get(Task, jid)
        if task is None:
            raise HTTPException(404, "任务不存在")
        return task

    @app.post("/v1/asr-jobs")
    def create_job(body: CreateJobReq, request: Request,
                   idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
                   tenant: str = Depends(require_tenant)):
        with tenant_session(request.app.state.db_engine, tenant) as db:
            session, _ = _get_or_create_session(db, tenant, body.external_id)

            def _dup() -> dict:
                dup = db.scalars(select(Task).where(Task.session_id == session.id,
                                                    Task.idem_key == idempotency_key)).first()
                return {"job_id": dup.id, "session_id": session.id,
                        "external_id": session.external_id,
                        "status": dup.status, "created": False, "duplicated": True}

            if idempotency_key:
                existing = db.scalars(select(Task.id).where(
                    Task.session_id == session.id, Task.idem_key == idempotency_key)).first()
                if existing:
                    return _dup()
            job_id = _new_id("job")
            try:
                with db.begin_nested():
                    db.add(Task(id=job_id, tenant_id=tenant, session_id=session.id,
                                status="pending_upload", idem_key=idempotency_key,
                                created_at=now_iso()))
            except IntegrityError:
                return _dup()
        return {"job_id": job_id, "session_id": session.id,
                "external_id": body.external_id,
                "status": "pending_upload", "created": True, "duplicated": False}

    @app.post("/v1/asr-jobs/{jid}/upload-authorization")
    def upload_authorization(jid: str, request: Request,
                             tenant: str = Depends(require_tenant)):
        with tenant_session(request.app.state.db_engine, tenant) as db:
            task = _get_job(db, jid)
            if task.status not in ("pending_upload", "uploaded"):
                raise HTTPException(409, "任务已进入转写流程，不能再上传")
            token = secrets.token_urlsafe(32)
            expires = datetime.now(timezone.utc) + timedelta(seconds=UPLOAD_TOKEN_TTL)
            task.upload_token = token
            task.upload_expires_at = expires.isoformat(timespec="microseconds")
        url = str(request.base_url).rstrip("/") + "/v1/uploads/" + token
        return {"job_id": jid, "upload_url": url, "method": "PUT",
                "expires_at": task.upload_expires_at}

    @app.put("/v1/uploads/{token}")
    async def put_upload(token: str, request: Request,
                         tenant: str = Depends(require_tenant)):
        cfg = request.app.state.config
        # 先短事务校验 token（RLS 下错租户天然查不到），再流式收字节，不占着数据库连接
        with tenant_session(request.app.state.db_engine, tenant) as db:
            task = db.scalars(select(Task).where(Task.upload_token == token)).first()
            if task is None:
                raise HTTPException(404, "上传地址无效")
            if task.status not in ("pending_upload", "uploaded"):
                raise HTTPException(409, "任务已进入转写流程，不能再上传")
            if task.upload_expires_at and \
                    datetime.fromisoformat(task.upload_expires_at) < datetime.now(timezone.utc):
                raise HTTPException(410, "上传地址已过期，请重新获取授权")
            job_id = task.id
        dest = tmp_dir(cfg, tenant) / f"{job_id}.src"
        limit = cfg.max_upload_mb * 1024 * 1024
        size = 0
        with open(dest, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"音频超过 {cfg.max_upload_mb}MB 上限")
                f.write(chunk)
        if size == 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "空文件")
        with tenant_session(request.app.state.db_engine, tenant) as db:
            task = db.get(Task, job_id)
            task.status = "uploaded"
            task.audio_path = str(dest)
        return {"job_id": job_id, "received_bytes": size, "status": "uploaded"}

    @app.post("/v1/asr-jobs/{jid}/upload-complete")
    def upload_complete(jid: str, request: Request, tenant: str = Depends(require_tenant)):
        with tenant_session(request.app.state.db_engine, tenant) as db:
            task = _get_job(db, jid)
            if task.status == "pending_upload":
                raise HTTPException(409, "尚未上传音频")
            if task.status != "uploaded":
                return {"job_id": jid, "session_id": task.session_id,
                        "status": task.status, "queued": False}
            seq = db.scalar(select(func.coalesce(func.max(Task.seq), 0) + 1)
                            .where(Task.session_id == task.session_id))
            task.seq = seq
            task.status = "queued"
            task.upload_token = None
            task.upload_expires_at = None
            session_id = task.session_id
        request.app.state.scheduler.enqueue(tenant, session_id, jid)
        return {"job_id": jid, "session_id": session_id, "seq": seq,
                "status": "queued", "queued": True}

    def _utterances_payload(db, session_id: str, rows) -> list[dict]:
        names = spk.name_map(db, session_id)
        out = []
        for r in rows:
            n = names.get(r.person_id, {"person_id": r.person_id,
                                        "person_name": "未知", "named": False})
            out.append({
                "cursor": r.id, "task_id": r.task_id, "seq": r.seq,
                "start_ms": r.start_ms, "end_ms": r.end_ms, "text": r.text,
                "person_id": n["person_id"], "person_name": n["person_name"],
                "named": n["named"],
            })
        return out

    @app.get("/v1/tasks/{tid}")
    def get_task(tid: str, request: Request, tenant: str = Depends(require_tenant)):
        with tenant_session(request.app.state.db_engine, tenant) as db:
            task = db.get(Task, tid)
            if task is None:
                raise HTTPException(404, "任务不存在")
            session = db.get(SessionRecord, task.session_id)
            body = {
                "task_id": task.id, "session_id": task.session_id, "seq": task.seq,
                "status": task.status, "error": task.error,
                "duration_ms": task.duration_ms, "created_at": task.created_at,
                "done_at": task.done_at,
                "session_version": session.version if session else None,
            }
            if task.status == "done":
                rows = db.scalars(select(Utterance).where(Utterance.task_id == tid)
                                  .order_by(Utterance.seq)).all()
                body["utterances"] = _utterances_payload(db, task.session_id, rows)
        return body

    @app.get("/v1/sessions/{sid}/tasks")
    def list_tasks(sid: str, request: Request, limit: int = 50,
                   tenant: str = Depends(require_tenant)):
        _get_session(request.app.state.db_engine, tenant, sid)
        limit = max(1, min(limit, 200))
        with tenant_session(request.app.state.db_engine, tenant) as db:
            rows = db.scalars(select(Task).where(Task.session_id == sid)
                              .order_by(Task.seq.desc()).limit(limit)).all()
        return {"session_id": sid, "tasks": [{
            "task_id": t.id, "seq": t.seq, "status": t.status, "error": t.error,
            "duration_ms": t.duration_ms, "created_at": t.created_at,
            "done_at": t.done_at, "idem_key": t.idem_key,
        } for t in rows]}

    @app.get("/v1/asr-jobs/{jid}")
    def get_job(jid: str, request: Request, tenant: str = Depends(require_tenant)):
        """轮询别名：job_id 即 task_id，响应与 GET /v1/tasks/{tid} 完全相同。"""
        return get_task(jid, request, tenant)

    @app.get("/v1/sessions/{sid}/results")
    def get_results(sid: str, request: Request, after: int = 0, limit: int = 500,
                    person_id: str | None = None,
                    tenant: str = Depends(require_tenant)):
        session = _get_session(request.app.state.db_engine, tenant, sid)
        limit = max(1, min(limit, 2000))
        with tenant_session(request.app.state.db_engine, tenant) as db:
            q = select(Utterance).where(Utterance.session_id == sid, Utterance.id > after)
            if person_id:
                names = spk.name_map(db, sid)
                if person_id not in names:
                    raise HTTPException(404, "说话人不存在")
                # 命中合并前的旧编号也算：按根人归拢后过滤
                pids = [p for p, n in names.items() if n["person_id"] == person_id]
                q = q.where(Utterance.person_id.in_(pids))
            rows = db.scalars(q.order_by(Utterance.id).limit(limit)).all()
            items = _utterances_payload(db, sid, rows)
        return {"session_id": sid, "session_version": session.version,
                "utterances": items,
                "next_cursor": items[-1]["cursor"] if items else after}

    @app.post("/v1/tasks/{tid}/retry")
    def retry_task(tid: str, request: Request, tenant: str = Depends(require_tenant)):
        with tenant_session(request.app.state.db_engine, tenant) as db:
            task = db.get(Task, tid)
            if task is None:
                raise HTTPException(404, "任务不存在")
            if task.status != "failed":
                raise HTTPException(409, "仅失败任务可重试")
            if not task.audio_path or not Path(task.audio_path).exists():
                raise HTTPException(409, "原始音频已不存在，请重新上传")
            task.status = "queued"
            task.error = None
        request.app.state.scheduler.enqueue(tenant, task.session_id, tid)
        return {"task_id": tid, "status": "queued"}

    # ---- 说话人与标签 ----

    def _speaker_op(db_engine, tenant, sid, fn):
        _get_session(db_engine, tenant, sid)
        try:
            with tenant_session(db_engine, tenant) as db:
                fn(db)
        except SpeakerError as e:
            raise HTTPException(e.status, e.detail)

    @app.get("/v1/sessions/{sid}/speakers")
    def list_speakers(sid: str, request: Request, tenant: str = Depends(require_tenant)):
        session = _get_session(request.app.state.db_engine, tenant, sid)
        with tenant_session(request.app.state.db_engine, tenant) as db:
            items = spk.list_speakers(db, sid)
        return {"session_id": sid, "session_version": session.version, "speakers": items}

    @app.get("/v1/sessions/{sid}/speakers/{pid}/tasks")
    def speaker_tasks(sid: str, pid: str, request: Request,
                      tenant: str = Depends(require_tenant)):
        session = _get_session(request.app.state.db_engine, tenant, sid)
        try:
            with tenant_session(request.app.state.db_engine, tenant) as db:
                body = spk.speaker_tasks(db, sid, pid)
        except SpeakerError as e:
            raise HTTPException(e.status, e.detail)
        return {"session_id": sid, "session_version": session.version, **body}

    @app.put("/v1/sessions/{sid}/speakers/{pid}/name")
    def set_name(sid: str, pid: str, body: NameReq, request: Request,
                 tenant: str = Depends(require_tenant)):
        try:
            name = spk.sanitize_name(body.name)
        except ValueError as e:
            raise HTTPException(422, str(e))
        _speaker_op(request.app.state.db_engine, tenant, sid,
                    lambda db: spk.set_name(db, sid, pid, name))
        return {"person_id": pid, "display_name": name}

    @app.post("/v1/sessions/{sid}/speakers/merge")
    def merge_speakers(sid: str, body: MergeReq, request: Request,
                       tenant: str = Depends(require_tenant)):
        _speaker_op(request.app.state.db_engine, tenant, sid,
                    lambda db: spk.merge(db, sid, body.from_person, body.into_person))
        return {"merged": True}

    @app.post("/v1/sessions/{sid}/speakers/{pid}/unmerge")
    def unmerge_speaker(sid: str, pid: str, request: Request,
                        tenant: str = Depends(require_tenant)):
        _speaker_op(request.app.state.db_engine, tenant, sid,
                    lambda db: spk.unmerge(db, sid, pid))
        return {"unmerged": True}

    # ---- 运维 ----

    @app.get("/console")
    def console():
        """内置测试台（页面本身无鉴权，接口调用仍需 API key）。"""
        return FileResponse(Path(__file__).parent / "static" / "console.html")

    @app.get("/panel")
    def panel():
        """说话人检视面板：说话人 → 录音 → 段落 三级下钻。"""
        return FileResponse(Path(__file__).parent / "static" / "panel.html")

    @app.get("/v1/health")
    def health(request: Request, tenant: str = Depends(require_tenant)):
        with tenant_session(request.app.state.db_engine, tenant) as db:
            row = db.execute(
                select(
                    func.count().filter(Task.status == "done"),
                    func.count().filter(Task.status == "failed"),
                    func.avg(Task.duration_ms).filter(Task.status == "done"),
                ).select_from(Task)
            ).one()
        avg_ms = int(row[2]) if row[2] is not None else None
        return {"status": "ok", "engine": request.app.state.engine.name,
                "queue": request.app.state.scheduler.stats(),
                "done": row[0], "failed": row[1], "avg_duration_ms": avg_ms}

    return app


app = create_app()
