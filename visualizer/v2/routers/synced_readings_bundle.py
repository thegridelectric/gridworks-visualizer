from datetime import datetime, timedelta
import math
from typing import Annotated, Dict, Self
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, model_validator

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session


from gw_data.db.models import (
    ReadingChannelSql,
    ReadingSql,
)

from sema_module.sema.types.synced_readings_bundle import ChannelReadingsListItem, SyncedReadingsBundle

from ..dependencies import get_db


router = APIRouter()

MAX_POINTS = 100

class ReadingsQueryParams(BaseModel):
    start: datetime
    end: datetime
    time_step: int | None = None
    channels: str = Field('')

    @model_validator(mode="after")
    def check_start_end(self) -> Self:
        if self.start >= self.end:
            raise ValueError("end_time must be after start_time")
        return self

    @model_validator(mode="after")
    def check_time_step(self) -> Self:
        if self.time_step and (self.end - self.start).total_seconds() / self.time_step > MAX_POINTS:
            raise ValueError("Too many points requested. Select a shorter time range or larger time step.")
        return self



DEFAULT_TIME_STEPS = [1,5,30,60,300,1200]

def datetime_to_sema(dt: datetime):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

@router.get('/api/v2/installations/{installation_id}/synced.readings.bundle')
def get_readings(installation_id, query: Annotated[ReadingsQueryParams, Query()], db: Session = Depends(get_db)):
    
    time_range_seconds = (query.end - query.start).total_seconds()
    time_step_seconds = query.time_step if query.time_step else next(i for i in DEFAULT_TIME_STEPS if i >= time_range_seconds / MAX_POINTS)
    query_interval = text(f"INTERVAL '{time_step_seconds} seconds'")

    channels = query.channels.split(',')
    ta_alias = installation_id + ".ta"

    # To get an accurate and complete set of time-averaged data for the requested time range,
    # our query needs to include the last value from before our time range begins.
    # Otherwise, data will be missing for any of our time steps that end before the timestamp of our first value.
    # Additionally, the first time step that actually does contain a value will not be able to compute an accurate 
    # average value, since it won't know its starting value.
    # 
    # We have no good way to know how far back to search, so we just go a single time step back and hope that it's enough.
    db_query_start = query.start - timedelta(seconds=time_step_seconds)

    # Additionally, we need to query for a full time step after our time range so that we can calculate the average value
    # of the time step that begins at the requested end time.
    db_query_end = query.end + timedelta(seconds=time_step_seconds)



		# SELECT 
		# 	reading_channels.name AS channel_name, 
		# 	reading_channels.unit AS channel_unit, 
		# 	reading_channels.unit_type AS channel_unit_type, 
		# 	time_bucket(INTERVAL '30 seconds', readings.timestamp) AS time_bucket, 
		# 	time_weight('LOCF', readings.timestamp, readings.value) AS time_weight
		# FROM readings 
		# JOIN reading_channels ON reading_channels.id = readings.channel_id
		# WHERE 
		# 	readings.timestamp >= '2026-01-02T00:00:00'
		# 	AND readings.timestamp <= '2026-01-02T00:05:00'
		# 	AND reading_channels.terminal_asset_alias = 'hw1.isone.me.versant.keene.beech.ta' 
		# 	AND reading_channels.name IN ('hp-ewt') 
		# GROUP BY time_bucket, reading_channels.name, reading_channels.unit, reading_channels.unit_type 
		# ORDER BY reading_channels.name, time_bucket


    time_weight_query = select(
        ReadingChannelSql.name.label('channel_name'),
        ReadingChannelSql.unit.label('channel_unit'),
        ReadingChannelSql.unit_type.label('channel_unit_type'),
        func.time_bucket(query_interval, ReadingSql.timestamp).label('time_bucket'),
        func.time_weight('LOCF', ReadingSql.timestamp, ReadingSql.value).label('time_weight')
    ).join(ReadingChannelSql).filter(
        ReadingSql.timestamp >= db_query_start,
        ReadingSql.timestamp <= db_query_end,
        ReadingChannelSql.terminal_asset_alias == ta_alias,
        ReadingChannelSql.name.in_(channels)
    ).group_by(
        text('time_bucket'),
        ReadingChannelSql.name,
        ReadingChannelSql.unit,
        ReadingChannelSql.unit_type
    ).order_by(
        ReadingChannelSql.name,
        text('time_bucket')
    ).subquery()

	# SELECT 
	# 	anon_2.channel_name AS channel_name, 
	# 	anon_2.channel_unit AS channel_unit, 
	# 	anon_2.channel_unit_type AS channel_unit_type, 
	# 	anon_2.time_bucket AS time_bucket,
	# 	interpolated_average(
	# 		anon_2.time_weight, 
	# 		time_bucket, 
	# 		INTERVAL '30 seconds', 
	# 		lag(anon_2.time_weight) OVER (PARTITION BY channel_name ORDER BY time_bucket), 
	# 		lead(anon_2.time_weight) OVER (PARTITION BY channel_name ORDER BY time_bucket)
	# 	) AS avg_value
	# FROM (
    #   -- time_weight_query
    # ) AS anon_2

    interpolated_avg_query = select(
        time_weight_query.c.channel_name.label('channel_name'),
        time_weight_query.c.channel_unit.label('channel_unit'),
        time_weight_query.c.channel_unit_type.label('channel_unit_type'),
        time_weight_query.c.time_bucket.label('time_bucket'),
        func.interpolated_average(
            time_weight_query.c.time_weight,
            text('time_bucket'),
            query_interval,
            func.lag(time_weight_query.c.time_weight).over(partition_by=text('channel_name'), order_by=text('time_bucket')),
            func.lead(time_weight_query.c.time_weight).over(partition_by=text('channel_name'), order_by=text('time_bucket')),
        ).label('avg_value')
    ).subquery()

    # SELECT 
    # 	anon_1.channel_name, 
    # 	anon_1.channel_unit, 
    # 	anon_1.channel_unit_type, 
    # 	time_bucket_gapfill(INTERVAL '30 seconds', anon_1.time_bucket) AS time_bucket_gapfilled,
    # 	locf(max(anon_1.avg_value)) AS locf_1
    # FROM (
    #   -- interpolated_avg_query
    # ) AS anon_1
    # WHERE
    # 	anon_1.avg_value IS NOT NULL
    # 	AND anon_1.time_bucket >= '2026-01-02T00:00:00'
    # 	AND anon_1.time_bucket <= '2026-01-02T00:05:00'        
    # GROUP BY 
    # 	anon_1.channel_name, anon_1.channel_unit, anon_1.channel_unit_type,	
    # 	time_bucket_gapfilled

    gapfilled_query = select(
        interpolated_avg_query.c.channel_name,
        interpolated_avg_query.c.channel_unit,
        interpolated_avg_query.c.channel_unit_type,
        func.time_bucket_gapfill(query_interval, interpolated_avg_query.c.time_bucket).label('time_bucket_gapfilled'),
        func.locf(func.max(interpolated_avg_query.c.avg_value))
    ).where(
        interpolated_avg_query.c.avg_value.is_not(None),
        interpolated_avg_query.c.time_bucket >= db_query_start,
        interpolated_avg_query.c.time_bucket <= db_query_end
    ).group_by(
        interpolated_avg_query.c.channel_name,
        interpolated_avg_query.c.channel_unit,
        interpolated_avg_query.c.channel_unit_type,
        text('time_bucket_gapfilled')
    )

    db_result = db.execute(gapfilled_query).all()


    result_bucket_count = math.floor(time_range_seconds / time_step_seconds) + 1
    db_result_bucket_count = result_bucket_count + 2

    # Our result is a list of rows of (name, time, value).
    # Each name will have time_count consecutive entries in ascending time order
    # The time values will be repeated for each name
    # We will have one extra piece of data at the front and the end, which 

    times = [datetime_to_sema(row[3]) for row in db_result[1:result_bucket_count+1]]

    channel_readings = []
    channel_count = len(db_result) / db_result_bucket_count
    for i in range(0, int(channel_count)):
        start_idx = i * db_result_bucket_count + 1
        channel_readings.append(ChannelReadingsListItem(
            channel_name=db_result[start_idx][0],
            unit=db_result[start_idx][1],
            unit_type=db_result[start_idx][2],
            value_list=[None if row[4] is None else int(row[4]) for row in db_result[start_idx:start_idx + result_bucket_count]]
        ))

    result = SyncedReadingsBundle(
        about_gnode_alias=ta_alias,
        start_timestamp=datetime_to_sema(query.start),
        end_timestamp=datetime_to_sema(query.end),
        timestamp_list=times,
        channel_readings_list=channel_readings
    )

    return result
