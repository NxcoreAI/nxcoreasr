"""跨线程任务事件：worker 线程发布 done/failed，WS 推送侧订阅（见 ws.py）。"""
import logging
from typing import Callable

log = logging.getLogger(__name__)

_subscribers: list[Callable[[str, str, str], None]] = []


def subscribe(fn: Callable[[str, str, str], None]) -> None:
    _subscribers.append(fn)


def publish(tenant: str, task_id: str, status: str) -> None:
    for fn in list(_subscribers):
        try:
            fn(tenant, task_id, status)
        except Exception:
            log.exception("任务事件订阅者异常 tenant=%s task=%s", tenant, task_id)
