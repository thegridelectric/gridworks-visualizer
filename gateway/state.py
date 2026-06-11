import re
import time
from dataclasses import dataclass, field
from typing import Optional

THERMOSTAT_CHANNEL_PATTERN = re.compile(
    r"^zone(?P<zone_number>\d+)-(?P<human_name>.*)-(temp|set|state)$"
)


def short_alias_from_gnode(g_node_alias: str) -> str | None:
    parts = g_node_alias.split(".")
    if len(parts) < 2:
        return None
    return parts[-2]


def thermostat_names_from_layout(layout: dict) -> list[str]:
    names: list[str] = []
    for channel in layout.get("DataChannels", []):
        match = THERMOSTAT_CHANNEL_PATTERN.match(channel.get("Name", ""))
        if match and (human_name := match.group("human_name")) not in names:
            names.append(human_name)
    return names


def empty_status_message(connected_clients: int) -> dict:
    return {
        "type": "status",
        "mqtt_connected": True,
        "layout_loaded": False,
        "snapshot_loaded": False,
        "target_gnode": "",
        "thermostat_names": [],
        "connected_clients": connected_clients,
    }


@dataclass
class HouseState:
    g_node_alias: str
    short_alias: str
    layout: Optional[dict] = None
    snapshot: Optional[dict] = None
    thermostat_names: list[str] = field(default_factory=list)
    messages_received: int = 0
    last_message_time: float = 0.0

    @property
    def snapshot_time_ms(self) -> int:
        if self.snapshot is None:
            return 0
        return int(self.snapshot.get("SnapshotTimeUnixMs", 0))

    def status_message(self, connected_clients: int) -> dict:
        last_activity = "Never"
        if self.last_message_time:
            last_activity = f"{int(time.time() - self.last_message_time)}s ago"
        return {
            "type": "status",
            "mqtt_connected": True,
            "layout_loaded": self.layout is not None,
            "snapshot_loaded": self.snapshot is not None,
            "target_gnode": self.g_node_alias,
            "thermostat_names": self.thermostat_names,
            "messages_received": self.messages_received,
            "connected_clients": connected_clients,
            "last_activity": last_activity,
        }

    def snapshot_message(self) -> Optional[dict]:
        if self.snapshot is None:
            return None
        return {
            "type": "mqtt_message",
            "message_type": "snapshot.spaceheat",
            "payload": self.snapshot,
        }


class HouseStateStore:
    def __init__(self) -> None:
        self._by_short_alias: dict[str, HouseState] = {}

    def get(self, short_alias: str) -> Optional[HouseState]:
        return self._by_short_alias.get(short_alias)

    def all_houses(self) -> list[HouseState]:
        return list(self._by_short_alias.values())

    def _get_or_create(self, g_node_alias: str) -> Optional[HouseState]:
        short_alias = short_alias_from_gnode(g_node_alias)
        if short_alias is None:
            return None
        state = self._by_short_alias.get(short_alias)
        if state is None:
            state = HouseState(g_node_alias=g_node_alias, short_alias=short_alias)
            self._by_short_alias[short_alias] = state
        return state

    def update_layout(self, g_node_alias: str, layout: dict) -> Optional[HouseState]:
        state = self._get_or_create(g_node_alias)
        if state is None:
            return None
        state.layout = layout
        state.thermostat_names = thermostat_names_from_layout(layout)
        state.messages_received += 1
        state.last_message_time = time.time()
        return state

    def update_snapshot(self, g_node_alias: str, snapshot: dict) -> Optional[HouseState]:
        state = self._get_or_create(g_node_alias)
        if state is None:
            return None
        snapshot_time = int(snapshot.get("SnapshotTimeUnixMs", 0))
        if state.snapshot is not None and snapshot_time <= state.snapshot_time_ms:
            return None
        state.snapshot = snapshot
        state.messages_received += 1
        state.last_message_time = time.time()
        return state
