"""AMQP consumer: one connection to RabbitMQ for all houses.

The SCADAs publish over the broker's MQTT plugin, which routes every message
into the `amq.topic` exchange. We bind an exclusive auto-delete queue with a
wildcard binding key, so the gateway observes all SCADA traffic without the
SCADAs (or LTNs) knowing it exists, and without leaving a backlog queue
behind when the gateway is down (a missed snapshot is replaced ~30s later by
the next one anyway).
"""

import asyncio
import json
import logging
from typing import Awaitable, Callable

import aio_pika

from gateway.config import GatewaySettings
from gateway.state import HouseState, HouseStateStore
from gateway.topics import decode_routing_key

logger = logging.getLogger(__name__)

LAYOUT_LITE = "layout.lite"
SNAPSHOT_SPACEHEAT = "snapshot.spaceheat"
HANDLED_MESSAGE_TYPES = {LAYOUT_LITE, SNAPSHOT_SPACEHEAT}

# Called with the updated house state whenever a house has fresh data to
# broadcast to its WebSocket clients.
HouseUpdateCallback = Callable[[HouseState], Awaitable[None]]


async def consume(
    settings: GatewaySettings,
    store: HouseStateStore,
    on_house_update: HouseUpdateCallback,
) -> None:
    """Run forever; connect_robust transparently reconnects on broker loss."""
    connection = await aio_pika.connect_robust(settings.rabbit_url.get_secret_value())
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=100)
        exchange = await channel.get_exchange(settings.rabbit_exchange, ensure=False)
        queue = await channel.declare_queue(exclusive=True, auto_delete=True)
        await queue.bind(exchange, routing_key=settings.rabbit_binding_key)
        logger.info(
            "Consuming from exchange %r with binding key %r",
            settings.rabbit_exchange,
            settings.rabbit_binding_key,
        )
        async with queue.iterator() as messages:
            async for message in messages:
                async with message.process():
                    try:
                        await _handle_message(
                            message.routing_key or "",
                            message.body,
                            store,
                            on_house_update,
                        )
                    except Exception:
                        logger.exception(
                            "Error handling message with routing key %r",
                            message.routing_key,
                        )


async def _handle_message(
    routing_key: str,
    body: bytes,
    store: HouseStateStore,
    on_house_update: HouseUpdateCallback,
) -> None:
    decoded = decode_routing_key(routing_key)
    if decoded is None:
        return
    # Only telemetry published by a SCADA is of interest.
    if not decoded.src.endswith(".scada"):
        return
    if decoded.message_type not in HANDLED_MESSAGE_TYPES:
        return

    envelope = json.loads(body)
    payload = envelope.get("Payload")
    if not isinstance(payload, dict):
        logger.warning("Message %r has no dict Payload", routing_key)
        return

    if decoded.message_type == LAYOUT_LITE:
        state = store.update_layout(decoded.src, payload)
    else:
        state = store.update_snapshot(decoded.src, payload)
    if state is not None:
        await on_house_update(state)


async def run_consumer_forever(
    settings: GatewaySettings,
    store: HouseStateStore,
    on_house_update: HouseUpdateCallback,
) -> None:
    """Restart the consumer if it ever exits with an error (connect_robust
    handles reconnects, but the initial connect can still fail)."""
    while True:
        try:
            await consume(settings, store, on_house_update)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Consumer failed; retrying in 5s")
            await asyncio.sleep(5)
