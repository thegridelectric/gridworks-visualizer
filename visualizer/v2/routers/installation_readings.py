from datetime import datetime
import math
from typing import Annotated, Dict
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from sqlalchemy import func, text
from sqlalchemy.orm import Session


from gw_data.db.models import (
    DataChannelSql,
    ReadingSql,
)

from ..dependencies import get_db


router = APIRouter()

class ReadingsQueryParams(BaseModel):
    start: datetime
    end: datetime
    channels: str = Field('')

class ReadingsResult(BaseModel):
    start: datetime
    end: datetime
    times: list[datetime]
    data: Dict[str, list[float]]

MAX_POINTS = 100
INTERVAL_OPTIONS_SECONDS = [1,5,30,60,300,1200]

@router.get('/api/v2/installations/{installation_id}/readings')
def get_readings(installation_id, query: Annotated[ReadingsQueryParams, Query()], db: Session = Depends(get_db)):
    
    time_range_seconds = (query.end - query.start).total_seconds()
    ideal_interval_seconds = time_range_seconds / MAX_POINTS
    interval_seconds = next(i for i in INTERVAL_OPTIONS_SECONDS if i >= ideal_interval_seconds)
    time_count = math.floor(time_range_seconds / interval_seconds) + 1
    query_interval = text(f"INTERVAL '{interval_seconds} seconds'")

    channels = query.channels.split(',')
    ta_alias = installation_id + ".ta"

    db_query = db.query(
        DataChannelSql.name,
        func.time_bucket_gapfill(query_interval, ReadingSql.timestamp).label('time'),
        func.locf(func.avg(ReadingSql.value)).label('avg_value')
    ).join(DataChannelSql).filter(
        ReadingSql.timestamp >= query.start,
        ReadingSql.timestamp <= query.end,
        DataChannelSql.terminal_asset_alias == ta_alias,
        DataChannelSql.name.in_(channels)
    ).group_by(
        text('time'),
        DataChannelSql.name
    ).order_by(
        DataChannelSql.name,
        text('time')
    )

    db_result = db_query.all()

    # Our result is a list of rows of (name, time, value).
    # Each name will have time_count consecutive entries in ascending time order
    # The time values will be repeated for each name

    times = [row[1] for row in db_result[0:time_count]]

    result = ReadingsResult(
        start=query.start,
        end=query.end,
        times= times,
        data = {}
    )

    channel_count = len(db_result) / time_count
    for i in range(0, int(channel_count)):
        start_idx = i * time_count
        result.data[db_result[start_idx][0]] = [float(row[2]) for row in db_result[start_idx:start_idx + time_count]]

    for ch in channels:
        if ch not in result.data:
            result.data[ch] = []

    return result
