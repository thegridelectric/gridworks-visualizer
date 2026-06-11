import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from gateway.config import GatewaySettings
from gateway.consumer import run_consumer_forever
from gateway.state import HouseState, HouseStateStore, empty_status_message

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: dict[str, set[WebSocket]] = {}

    def add(self, short_alias: str, websocket: WebSocket) -> None:
        self._clients.setdefault(short_alias, set()).add(websocket)

    def remove(self, short_alias: str, websocket: WebSocket) -> None:
        clients = self._clients.get(short_alias)
        if clients is not None:
            clients.discard(websocket)
            if not clients:
                del self._clients[short_alias]

    def client_count(self, short_alias: str) -> int:
        return len(self._clients.get(short_alias, ()))

    def total_clients(self) -> int:
        return sum(len(clients) for clients in self._clients.values())

    async def broadcast(self, short_alias: str, messages: list[dict]) -> None:
        clients = list(self._clients.get(short_alias, ()))
        if not clients:
            return
        encoded = [json.dumps(message) for message in messages]
        for websocket in clients:
            try:
                for text in encoded:
                    await websocket.send_text(text)
            except Exception:
                logger.debug("Failed to send to a client of %r", short_alias)


def create_app(settings: GatewaySettings) -> FastAPI:
    store = HouseStateStore()
    manager = ConnectionManager()
    started_at = time.time()

    async def on_house_update(state: HouseState) -> None:
        messages = [state.status_message(manager.client_count(state.short_alias))]
        if (snapshot_message := state.snapshot_message()) is not None:
            messages.append(snapshot_message)
        await manager.broadcast(state.short_alias, messages)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        consumer_task = asyncio.create_task(
            run_consumer_forever(settings, store, on_house_update)
        )
        try:
            yield
        finally:
            consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer_task

    app = FastAPI(title="GridWorks Realtime Gateway", lifespan=lifespan)

    async def health() -> dict:
        return {
            "status": "ok",
            "uptime_seconds": int(time.time() - started_at),
            "connected_clients": manager.total_clients(),
            "houses": [
                {
                    "short_alias": state.short_alias,
                    "g_node_alias": state.g_node_alias,
                    "layout_loaded": state.layout is not None,
                    "snapshot_loaded": state.snapshot is not None,
                    "snapshot_time_unix_ms": state.snapshot_time_ms,
                    "messages_received": state.messages_received,
                    "connected_clients": manager.client_count(state.short_alias),
                }
                for state in store.all_houses()
            ],
        }

    async def send_cached(websocket: WebSocket, short_alias: str) -> None:
        state = store.get(short_alias)
        if state is None:
            await websocket.send_text(
                json.dumps(empty_status_message(manager.client_count(short_alias)))
            )
            return
        await websocket.send_text(
            json.dumps(state.status_message(manager.client_count(short_alias)))
        )
        if (snapshot_message := state.snapshot_message()) is not None:
            await websocket.send_text(json.dumps(snapshot_message))

    async def websocket_endpoint(websocket: WebSocket, house_alias: str) -> None:
        await websocket.accept()
        manager.add(house_alias, websocket)
        logger.info(
            "Client connected to %r (%d clients)",
            house_alias,
            manager.client_count(house_alias),
        )
        try:
            await send_cached(websocket, house_alias)
            while True:
                text = await websocket.receive_text()
                try:
                    request = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if request.get("type") in ("get_status", "request_snapshot"):
                    await send_cached(websocket, house_alias)
        except WebSocketDisconnect:
            pass
        finally:
            manager.remove(house_alias, websocket)
            logger.info(
                "Client disconnected from %r (%d clients)",
                house_alias,
                manager.client_count(house_alias),
            )

    app.add_api_route("/gateway/health", health, methods=["GET"])
    app.add_api_websocket_route("/realtime/{house_alias}", websocket_endpoint)

    return app
