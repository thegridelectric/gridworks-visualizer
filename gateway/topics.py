"""Decoding of AMQP routing keys produced by the RabbitMQ MQTT plugin.

SCADAs publish MQTT topics of the form (see gwproto.MQTTTopic):

    gw/<src-gnode-alias>/to/<dst>/<message-type>

where '.' in the gnode alias and message type is replaced by '-'. The MQTT
plugin maps '/' to '.', so on the AMQP side the routing key looks like:

    gw.hw1-isone-me-versant-keene-oak-scada.to.ltn.snapshot-spaceheat
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedRoutingKey:
    envelope_type: str
    src: str  # gnode alias with dots restored, e.g. hw1.isone.me.versant.keene.oak.scada
    dst: str
    message_type: str  # with dots restored, e.g. snapshot.spaceheat


def decode_routing_key(routing_key: str) -> DecodedRoutingKey | None:
    """Decode an AMQP routing key into its gw-topic components.

    Returns None for keys that don't follow the ENVELOPE/SRC/to/DST/TYPE shape.
    """
    parts = routing_key.split(".")
    if len(parts) < 5 or parts[2] != "to":
        return None
    return DecodedRoutingKey(
        envelope_type=parts[0],
        src=parts[1].replace("-", "."),
        dst=parts[3].replace("-", "."),
        message_type=parts[4].replace("-", "."),
    )


def short_alias_from_gnode(g_node_alias: str) -> str | None:
    """hw1.isone.me.versant.keene.oak.scada -> oak

    Same rule used to derive the house short alias from its g-node path.
    """
    parts = g_node_alias.split(".")
    if len(parts) < 2:
        return None
    return parts[-2]
