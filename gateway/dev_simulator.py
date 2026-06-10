"""Dev tool: pretend to be one or more SCADAs publishing to RabbitMQ.

Publishes layout.lite once and then snapshot.spaceheat every few seconds,
with the same routing keys the broker's MQTT plugin would produce, so the
gateway can be exercised without any real SCADA.

Usage (from the repo root, against a local broker):

    python -m gateway.dev_simulator --houses oak,fir --interval 5
"""

import argparse
import asyncio
import json
import random
import time

import aio_pika

from gateway.config import GatewaySettings

ZONE_NAMES = ["living-rm", "upstairs"]


def routing_key(src_gnode: str, message_type: str) -> str:
    src = src_gnode.replace(".", "-")
    return f"gw.{src}.to.ltn.{message_type.replace('.', '-')}"


def fake_layout(g_node_alias: str) -> dict:
    channels = []
    for i, zone in enumerate(ZONE_NAMES, start=1):
        for suffix in ("temp", "set", "state"):
            channels.append({"Name": f"zone{i}-{zone}-{suffix}"})
    channels += [{"Name": "hp-odu-pwr"}, {"Name": "hp-idu-pwr"}, {"Name": "primary-flow"}]
    return {
        "TypeName": "layout.lite",
        "FromGNodeAlias": g_node_alias,
        "DataChannels": channels,
        "ShNodes": [],
    }


def fake_snapshot(g_node_alias: str) -> dict:
    now_ms = int(time.time() * 1000)
    readings = [
        {"ChannelName": "hp-odu-pwr", "Value": random.randint(0, 4000), "ScadaReadTimeUnixMs": now_ms},
        {"ChannelName": "hp-idu-pwr", "Value": random.randint(0, 800), "ScadaReadTimeUnixMs": now_ms},
        {"ChannelName": "primary-flow", "Value": random.randint(0, 600), "ScadaReadTimeUnixMs": now_ms},
    ]
    for i, zone in enumerate(ZONE_NAMES, start=1):
        readings.append(
            {"ChannelName": f"zone{i}-{zone}-temp", "Value": random.randint(65000, 72000), "ScadaReadTimeUnixMs": now_ms}
        )
        readings.append(
            {"ChannelName": f"zone{i}-{zone}-set", "Value": 70000, "ScadaReadTimeUnixMs": now_ms}
        )
    return {
        "TypeName": "snapshot.spaceheat",
        "FromGNodeAlias": g_node_alias,
        "FromGNodeInstanceId": "00000000-0000-0000-0000-000000000000",
        "SnapshotTimeUnixMs": now_ms,
        "LatestReadingList": readings,
        "LatestStateList": [],
    }


def envelope(src: str, message_type: str, payload: dict) -> bytes:
    return json.dumps(
        {
            "Header": {"Src": src, "Dst": "ltn", "MessageType": message_type},
            "Payload": payload,
            "TypeName": "gw",
        }
    ).encode()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--houses", default="oak", help="comma-separated short aliases")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between snapshots")
    args = parser.parse_args()

    settings = GatewaySettings()
    houses = [alias.strip() for alias in args.houses.split(",") if alias.strip()]
    gnodes = {alias: f"hw1.isone.me.versant.keene.{alias}.scada" for alias in houses}

    connection = await aio_pika.connect_robust(settings.rabbit_url.get_secret_value())
    async with connection:
        channel = await connection.channel()
        exchange = await channel.get_exchange(settings.rabbit_exchange, ensure=False)

        for alias, gnode in gnodes.items():
            await exchange.publish(
                aio_pika.Message(body=envelope(gnode, "layout.lite", fake_layout(gnode))),
                routing_key=routing_key(gnode, "layout.lite"),
            )
            print(f"published layout.lite for {alias}")

        while True:
            for alias, gnode in gnodes.items():
                await exchange.publish(
                    aio_pika.Message(body=envelope(gnode, "snapshot.spaceheat", fake_snapshot(gnode))),
                    routing_key=routing_key(gnode, "snapshot.spaceheat"),
                )
                print(f"published snapshot.spaceheat for {alias}")
            await asyncio.sleep(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
