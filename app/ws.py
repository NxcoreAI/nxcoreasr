"""任务状态推送（WS）：长连接客户端订阅任务 done/failed，到点即推。

worker 线程在 service.process_task 落库后调 events.publish → hub.publish
经 run_coroutine_threadsafe 桥回事件循环广播。推送只带 task_id + status；
要完整结果走既有 REST（GET /v1/tasks 或 /v1/tasks?ids=）。
"""
import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)

MAX_SUBSCRIBED_IDS = 500


class TaskHub:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._clients: list[dict] = []

    def add(self, client: dict) -> None:
        self._clients.append(client)

    def remove(self, client: dict) -> None:
        if client in self._clients:
            self._clients.remove(client)

    def publish(self, tenant: str, task_id: str, status: str) -> None:
        """线程安全入口：worker 线程调用，转发到事件循环。"""
        if self.loop is None or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(tenant, task_id, status), self.loop)

    async def _broadcast(self, tenant: str, task_id: str, status: str) -> None:
        for client in list(self._clients):
            if client["tenant"] != tenant:
                continue
            if client["ids"] is not None and task_id not in client["ids"]:
                continue
            await client["queue"].put({"type": "task", "task_id": task_id, "status": status})


def _api_key_from(websocket: WebSocket) -> str | None:
    key = websocket.query_params.get("api_key")
    if key:
        return key
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


async def ws_tasks(websocket: WebSocket) -> None:
    key = _api_key_from(websocket)
    tenant = websocket.app.state.config.tenant_by_key(key) if key else None
    if tenant is None:
        await websocket.accept()
        await websocket.close(code=4401, reason="凭证无效")
        return
    await websocket.accept()
    hub: TaskHub = websocket.app.state.task_hub
    client = {"ws": websocket, "tenant": tenant.name, "ids": None, "queue": asyncio.Queue()}
    hub.add(client)
    writer = asyncio.create_task(_writer(client))
    try:
        while True:
            frame = json.loads(await websocket.receive_text())
            action = frame.get("action")
            if action == "subscribe":
                ids = sorted({i for i in frame.get("task_ids", []) if isinstance(i, str)})
                client["ids"] = set(ids[:MAX_SUBSCRIBED_IDS])
            elif action == "subscribe_all":
                client["ids"] = None
            elif action == "unsubscribe":
                client["ids"] = set()
            else:
                continue
            await client["queue"].put({"type": "subscribed",
                                       "mode": "all" if client["ids"] is None else "ids"})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("WS 任务推送连接异常 tenant=%s", tenant.name)
    finally:
        hub.remove(client)
        writer.cancel()


async def _writer(client: dict) -> None:
    try:
        while True:
            payload = await client["queue"].get()
            await client["ws"].send_text(json.dumps(payload))
    except Exception:
        pass  # 连接已断，读循环负责清理
