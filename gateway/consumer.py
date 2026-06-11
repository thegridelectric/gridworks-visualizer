import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

import aio_pika

from gateway.config import GatewaySettings
from gateway.state import HouseState, HouseStateStore

logger = logging.getLogger(__name__)

LAYOUT_LITE = "layout.lite"
SNAPSHOT_SPACEHEAT = "snapshot.spaceheat"
HANDLED_MESSAGE_TYPES = {LAYOUT_LITE, SNAPSHOT_SPACEHEAT}

HouseUpdateCallback = Callable[[HouseState], Awaitable[None]]


@dataclass(frozen=True)
class DecodedRoutingKey:
    envelope_type: str
    src: str
    dst: str
    message_type: str


def decode_routing_key(routing_key: str) -> DecodedRoutingKey | None:
    """Decode an AMQP routing key into its gw-topic components."""
    parts = routing_key.split(".")
    if len(parts) < 5 or parts[2] != "to":
        return None
    return DecodedRoutingKey(
        envelope_type=parts[0],
        src=parts[1].replace("-", "."),
        dst=parts[3].replace("-", "."),
        message_type=parts[4].replace("-", "."),
    )


async def consume(
    settings: GatewaySettings,
    store: HouseStateStore,
    on_house_update: HouseUpdateCallback,
) -> None:
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
    if not decoded.src.endswith(".scada"):
        return
    if decoded.message_type not in HANDLED_MESSAGE_TYPES:
        return

    message = json.loads(body)
    payload = message.get("Payload")
    if not isinstance(payload, dict):
        logger.warning("Message %r has no dict Payload", routing_key)
        return

    if decoded.message_type == LAYOUT_LITE:
        state = store.update_layout(decoded.src, payload)
    elif decoded.message_type == SNAPSHOT_SPACEHEAT:
        state = store.update_snapshot(decoded.src, payload)
    else:
        return
    if state is not None:
        await on_house_update(state)


async def run_consumer_forever(
    settings: GatewaySettings,
    store: HouseStateStore,
    on_house_update: HouseUpdateCallback,
) -> None:
    while True:
        try:
            await consume(settings, store, on_house_update)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Consumer failed; retrying in 5s")
            await asyncio.sleep(5)
