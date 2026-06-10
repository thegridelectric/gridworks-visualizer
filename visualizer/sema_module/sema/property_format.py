import re
import uuid
from datetime import UTC, datetime
from typing import Annotated

from pydantic import BeforeValidator


# --- patterns ---
HANDLE_NAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*(?:\.[a-z][a-z0-9]*(?:-[a-z0-9]+)*)*$"
)

LEFT_RIGHT_DOT_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(\.[a-z0-9]+)*$"
)

SPACEHEAT_NAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"
)

UTC_ISO8601_SECONDS_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)

UUID4_STR_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


# --- methods ---
def is_handle_name(v: str) -> str:
    if not isinstance(v, str):
        raise ValueError(f"<{v}>: HandleName must be a string.")

    if not HANDLE_NAME_PATTERN.fullmatch(v):
        raise ValueError(f"<{v}>: Fails HandleName format.")

    return v


def is_left_right_dot(v: str) -> str:
    if not isinstance(v, str):
        raise ValueError(f"<{v}>: LeftRightDot must be a string.")

    if not LEFT_RIGHT_DOT_PATTERN.fullmatch(v):
        raise ValueError(f"<{v}>: Fails LeftRightDot format.")

    return v


def is_spaceheat_name(v: str) -> str:
    if not isinstance(v, str):
        raise ValueError(f"<{v}>: SpaceheatName must be a string.")

    if len(v) > 64:
        raise ValueError(f"<{v}>: SpaceheatName exceeds maximum length of 64.")

    if not SPACEHEAT_NAME_PATTERN.fullmatch(v):
        raise ValueError(f"<{v}>: Fails SpaceheatName format.")

    return v


def is_utc_iso8601_seconds(v: str) -> str:
    if not isinstance(v, str):
        raise ValueError(f"<{v}>: utc.iso8601.seconds must be a string.")

    if not UTC_ISO8601_SECONDS_PATTERN.fullmatch(v):
        raise ValueError(f"<{v}>: Fails utc.iso8601.seconds format.")

    return v


def is_utc_milliseconds(v: int) -> int:
    if not isinstance(v, int):
        raise TypeError("Not an int!")
    start_date = datetime(2000, 1, 1, tzinfo=UTC)
    end_date = datetime(3000, 1, 1, tzinfo=UTC)

    start_timestamp_ms = int(start_date.timestamp() * 1000)
    end_timestamp_ms = int(end_date.timestamp() * 1000)

    if v < start_timestamp_ms:
        raise ValueError(f"{v} must be after Jan 1 2000")
    if v > end_timestamp_ms:
        raise ValueError(f"{v} must be before Jan 1 3000")
    return v


def is_utc_seconds(v: int) -> int:
    if not isinstance(v, int):
        raise ValueError("Not an int!")
    start_date = datetime(2000, 1, 1, tzinfo=UTC)
    end_date = datetime(3000, 1, 1, tzinfo=UTC)

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    if v < start_timestamp:
        raise ValueError(f"{v}: Fails UTCSeconds format! Must be after Jan 1 2000")
    if v > end_timestamp:
        raise ValueError(f"{v}: Fails UTCSeconds format! Must be before Jan 1 3000")
    return v


def is_uuid4_str(v: str) -> str:
    if not isinstance(v, str):
        raise ValueError(f"<{v}>: uuid4.str must be a string.")

    if not UUID4_STR_PATTERN.fullmatch(v):
        raise ValueError(f"<{v}>: Fails uuid4.str format.")

    try:
        u = uuid.UUID(v)
    except Exception as e:
        raise ValueError(f"Invalid UUID4: {v}  <{e}>") from e
    if u.version != 4:
        raise ValueError(
            f"{v} is valid uid, but of version {u.version}. Fails UuidCanonicalTextual"
        )
    return str(u)


# --- annotated types ---
HandleName = Annotated[
    str,
    BeforeValidator(is_handle_name),
]

LeftRightDot = Annotated[
    str,
    BeforeValidator(is_left_right_dot),
]

SpaceheatName = Annotated[
    str,
    BeforeValidator(is_spaceheat_name),
]

UtcIso8601Seconds = Annotated[
    str,
    BeforeValidator(is_utc_iso8601_seconds),
]

UTCMilliseconds = Annotated[
    int,
    BeforeValidator(is_utc_milliseconds),
]

UTCSeconds = Annotated[
    int,
    BeforeValidator(is_utc_seconds),
]

UUID4Str = Annotated[
    str,
    BeforeValidator(is_uuid4_str),
]


# --- helpers ---
class UtcIso8601SecondsFormat:
    @staticmethod
    def from_datetime(dt: datetime) -> UtcIso8601Seconds:
        if not isinstance(dt, datetime):
            raise TypeError(f"{dt} must be a datetime")

        if dt.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")

        dt_utc = dt.astimezone(UTC)
        dt_utc = dt_utc.replace(microsecond=0)
        s = dt_utc.isoformat().replace("+00:00", "Z")

        return is_utc_iso8601_seconds(s)
