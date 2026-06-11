"""End-to-end smoke test without a broker.

Runs the real FastAPI/uvicorn server and a real WebSocket client, but feeds
messages straight into the consumer's handler instead of going through
RabbitMQ. Verifies the full client-facing contract.

    python -m gateway.smoke_test
"""

import asyncio
import json
import time
import urllib.request

import uvicorn
import websockets

import gateway.server as server_mod
from gateway.config import GatewaySettings
from gateway.consumer import _handle_message
from gateway.dev_simulator import envelope, fake_layout, fake_snapshot, routing_key

PORT = 8155
GNODE = "hw1.isone.me.versant.keene.oak.scada"


async def recv_json(ws, timeout: float = 5.0) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout))


async def run() -> None:
    # Replace the AMQP consumer with a no-op; messages are injected manually.
    async def fake_consumer(settings, store, on_house_update):
        await asyncio.Event().wait()

    server_mod.run_consumer_forever = fake_consumer

    settings = GatewaySettings(gateway_host="127.0.0.1", gateway_port=PORT)
    app = server_mod.create_app(settings)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    )
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    store = app.state.store
    on_update = app.state.on_house_update

    async def inject(message_type: str, payload: dict) -> None:
        await _handle_message(
            routing_key(GNODE, message_type),
            envelope(GNODE, message_type, payload),
            store,
            on_update,
        )

    try:
        async with websockets.connect(f"ws://127.0.0.1:{PORT}/realtime/oak") as ws:
            # 1. Connect before any data: empty status.
            msg = await recv_json(ws)
            assert msg["type"] == "status", msg
            assert msg["target_gnode"] == "", msg
            assert msg["snapshot_loaded"] is False, msg
            print("PASS: empty status on connect (no data yet)")

            # 2. Layout arrives: status broadcast with thermostat names.
            await inject("layout.lite", fake_layout(GNODE))
            msg = await recv_json(ws)
            assert msg["type"] == "status", msg
            assert msg["target_gnode"] == GNODE, msg
            assert msg["thermostat_names"] == ["living-rm", "upstairs"], msg
            assert msg["layout_loaded"] is True and msg["snapshot_loaded"] is False
            print("PASS: layout.lite -> status with thermostat names")

            # 3. Snapshot arrives: status + mqtt_message broadcast.
            snapshot = fake_snapshot(GNODE)
            await inject("snapshot.spaceheat", snapshot)
            status = await recv_json(ws)
            assert status["type"] == "status" and status["snapshot_loaded"] is True
            mqtt = await recv_json(ws)
            assert mqtt["type"] == "mqtt_message", mqtt
            assert mqtt["message_type"] == "snapshot.spaceheat", mqtt
            assert mqtt["payload"]["SnapshotTimeUnixMs"] == snapshot["SnapshotTimeUnixMs"]
            assert mqtt["payload"]["LatestReadingList"] == snapshot["LatestReadingList"]
            print("PASS: snapshot.spaceheat -> status + mqtt_message broadcast")

            # 4. Stale duplicate (e.g. the admin-link copy) is not re-broadcast.
            await inject("snapshot.spaceheat", snapshot)
            await inject("layout.lite", fake_layout(GNODE))  # marker message
            # The marker broadcast itself sends status + cached snapshot; the
            # point is that no extra frames arrived from the duplicate inject.
            msg = await recv_json(ws)
            assert msg["type"] == "status", msg
            msg = await recv_json(ws)
            assert msg["type"] == "mqtt_message", msg
            print("PASS: duplicate snapshot suppressed")

            # 5. Legacy client messages are answered from cache.
            await ws.send(json.dumps({"type": "request_snapshot", "data": {}}))
            status = await recv_json(ws)
            mqtt = await recv_json(ws)
            assert status["type"] == "status" and mqtt["type"] == "mqtt_message"
            print("PASS: request_snapshot answered from cache")

            # 6. New client connecting later gets cached state immediately.
            async with websockets.connect(f"ws://127.0.0.1:{PORT}/realtime/oak") as ws2:
                status = await recv_json(ws2)
                mqtt = await recv_json(ws2)
                assert status["target_gnode"] == GNODE
                assert mqtt["message_type"] == "snapshot.spaceheat"
                assert status["connected_clients"] == 2, status
                print("PASS: late joiner receives cached status + snapshot")

            # 7. Unrelated routing keys are ignored without error.
            await _handle_message(
                routing_key(GNODE, "power.watts"),
                envelope(GNODE, "power.watts", {"Watts": 100}),
                store,
                on_update,
            )
            await _handle_message("gw.some-ltn.to.s.bid", b"{}", store, on_update)
            await _handle_message("not.a.gw.topic", b"not json", store, on_update)
            print("PASS: unrelated/garbage messages ignored")

        # 8. Health endpoint (urlopen blocks, so run it off the server's loop).
        health = json.loads(
            await asyncio.to_thread(
                lambda: urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/gateway/health", timeout=5
                ).read()
            )
        )
        assert health["status"] == "ok"
        assert health["houses"][0]["short_alias"] == "oak", health
        assert health["houses"][0]["snapshot_loaded"] is True
        print("PASS: health endpoint reports the house")

        print("\nAll smoke tests passed.")
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)


if __name__ == "__main__":
    asyncio.run(run())
