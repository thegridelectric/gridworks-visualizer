from datetime import datetime
import math
from typing import Annotated, Dict, Self
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, model_validator

from sqlalchemy import func, text
from sqlalchemy.orm import Session


from gw_data.db.models import (
    ReadingChannelSql,
    ReadingSql,
)

from sema_module.sema.types.synced_readings_bundle import ChannelReadingsListItem, SyncedReadingsBundleGt

from ..dependencies import get_db


router = APIRouter()

MAX_POINTS = 100

class ReadingsQueryParams(BaseModel):
    start: datetime
    end: datetime
    interval: int | None = None
    channels: str = Field('')

    @model_validator(mode="after")
    def check_start_end(self) -> Self:
        if self.start >= self.end:
            raise ValueError("end_time must be after start_time")
        return self

    @model_validator(mode="after")
    def check_interval(self) -> Self:
        if self.interval and (self.end - self.start).total_seconds() / self.interval > MAX_POINTS:
            raise ValueError("Too many points requested. Select a shorter time range or larger interval.")
        return self



INTERVAL_OPTIONS_SECONDS = [1,5,30,60,300,1200]

def datetime_to_sema(dt: datetime):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

@router.get('/api/v2/installations/{installation_id}/synced.readings.bundle')
def get_readings(installation_id, query: Annotated[ReadingsQueryParams, Query()], db: Session = Depends(get_db)):
    
    time_range_seconds = (query.end - query.start).total_seconds()
    interval_seconds = query.interval if query.interval else next(i for i in INTERVAL_OPTIONS_SECONDS if i >= time_range_seconds / MAX_POINTS)
    time_count = math.floor(time_range_seconds / interval_seconds) + 1
    query_interval = text(f"INTERVAL '{interval_seconds} seconds'")

    channels = query.channels.split(',')
    ta_alias = installation_id + ".ta"

    db_query = db.query(
        ReadingChannelSql.name,
        ReadingChannelSql.unit,
        ReadingChannelSql.unit_type,
        func.time_bucket_gapfill(query_interval, ReadingSql.timestamp).label('time'),
        func.locf(func.avg(ReadingSql.value)).label('avg_value')
    ).join(ReadingChannelSql).filter(
        ReadingSql.timestamp >= query.start,
        ReadingSql.timestamp <= query.end,
        ReadingChannelSql.terminal_asset_alias == ta_alias,
        ReadingChannelSql.name.in_(channels)
    ).group_by(
        text('time'),
        ReadingChannelSql.name,
        ReadingChannelSql.unit,
        ReadingChannelSql.unit_type
    ).order_by(
        ReadingChannelSql.name,
        text('time')
    )

    db_result = db_query.all()

    # Our result is a list of rows of (name, time, value).
    # Each name will have time_count consecutive entries in ascending time order
    # The time values will be repeated for each name

    times = [datetime_to_sema(row[3]) for row in db_result[0:time_count]]

    channel_readings = []
    channel_count = len(db_result) / time_count
    for i in range(0, int(channel_count)):
        start_idx = i * time_count
        channel_readings.append(ChannelReadingsListItem(
            channel_name=db_result[start_idx][0],
            unit=db_result[start_idx][1],
            unit_type=db_result[start_idx][2],
            value_list=[None if row[4] is None else int(row[4]) for row in db_result[start_idx:start_idx + time_count]]
        ))

    result = SyncedReadingsBundleGt(
        about_gnode_alias=ta_alias,
        start_timestamp=datetime_to_sema(query.start),
        end_timestamp=datetime_to_sema(query.end),
        timestamp_list=times,
        channel_readings_list=channel_readings
    )

    return result
