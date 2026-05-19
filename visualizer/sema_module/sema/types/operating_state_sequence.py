from typing import Literal
from sema_module.sema.base import SemaType
from sema_module.sema.property_format import SpaceheatName
from sema_module.sema.property_format import UtcIso8601Seconds


class OperatingStateSequence(SemaType):
    """Sema: https://schemas.electricity.works/types/operating.state.sequence/000"""

    channel_name: SpaceheatName
    value_list: list[str | None]
    timestamp_list: list[UtcIso8601Seconds]
    type_name: Literal["operating.state.sequence"] = "operating.state.sequence"
    version: Literal["000"] = "000"
