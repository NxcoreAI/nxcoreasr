"""进程内持久化队列：同会话严格串行（按 seq），跨会话并行，全局并发上限。

任务真身在 SQLite；这里只管调度顺序。启动时把 queued/processing 的
任务恢复入队（processing 说明上次进程挂在半路，重置为 queued）。
"""
import logging
import threading
from collections import deque

from sqlalchemy import select, update

from . import service
from .config import AppConfig
from .db import Task, tenant_session
from .engine.base import Engine

log = logging.getLogger(__name__)

SessionKey = tuple[str, str]  # (tenant, session_id)


class Scheduler:
    def __init__(self, cfg: AppConfig, engine: Engine, db_engine):
        self.cfg = cfg
        self.engine = engine
        self.db_engine = db_engine
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: dict[SessionKey, deque[str]] = {}
        self._active: set[SessionKey] = set()
        self._stop = False
        self._threads: list[threading.Thread] = []

    # ---- 生命周期 ----

    def start(self) -> None:
        self._recover()
        for i in range(self.cfg.max_workers):
            t = threading.Thread(target=self._worker, name=f"asr-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout=5)

    def _recover(self) -> None:
        # RLS 不允许跨租户一把捞，按租户循环
        for tenant in [t.name for t in self.cfg.tenants]:
            with tenant_session(self.db_engine, tenant) as db:
                db.execute(update(Task).where(Task.status == "processing")
                           .values(status="queued"))
                rows = db.execute(
                    select(Task.id, Task.session_id)
                    .where(Task.status == "queued")
                    .order_by(Task.session_id, Task.seq)
                ).all()
            for r in rows:
                self.enqueue(tenant, r[1], r[0])
            if rows:
                log.info("恢复未完成任务 tenant=%s count=%d", tenant, len(rows))

    # ---- 调度 ----

    def enqueue(self, tenant: str, session_id: str, task_id: str) -> None:
        key = (tenant, session_id)
        with self._cond:
            self._pending.setdefault(key, deque()).append(task_id)
            self._cond.notify()

    def cancel_session(self, tenant: str, session_id: str, timeout: float = 300) -> bool:
        """删除会话前调用：丢弃排队任务，等在跑的收尾。返回是否等到。"""
        key = (tenant, session_id)
        with self._cond:
            self._pending.pop(key, None)
            return self._cond.wait_for(lambda: key not in self._active, timeout=timeout)

    def stats(self) -> dict:
        with self._lock:
            return {
                "queued": sum(len(d) for d in self._pending.values()),
                "processing": len(self._active),
                "workers": self.cfg.max_workers,
            }

    def _next(self) -> tuple[SessionKey, str] | None:
        for key, dq in self._pending.items():
            if key not in self._active and dq:
                task_id = dq.popleft()
                if not dq:
                    del self._pending[key]
                self._active.add(key)
                return key, task_id
        return None

    def _worker(self) -> None:
        while True:
            with self._cond:
                item = self._next()
                while item is None and not self._stop:
                    self._cond.wait()
                    item = self._next()
                if self._stop and item is None:
                    return
            key, task_id = item
            tenant, _session_id = key
            try:
                service.process_task(self.cfg, self.engine, self.db_engine, tenant, task_id)
            except Exception:
                log.exception("worker 异常 tenant=%s task=%s", tenant, task_id)
            finally:
                with self._cond:
                    self._active.discard(key)
                    self._cond.notify_all()
