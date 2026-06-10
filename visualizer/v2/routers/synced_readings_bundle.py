from datetime import datetime, timedelta
import math
from typing import Annotated, Dict, Self
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, model_validator

from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session


from gw_data.db.models import (
    MessageSql,
    ReadingChannelSql,
    ReadingSql,
)

from sema_module.sema.enums.gw_str_enum import SemaEnum
from sema_module.sema import enums as sema_enums
from sema_module.sema.types import (
    ChannelReadingsListItem,
    OperatingStateSequence,
    SyncedReadingsBundle,
)

from ..dependencies import get_db

SEMA_ENUM_LOOKUP: dict[str, SemaEnum] = {
    enum_class.enum_name(): enum_class
    for enum_class in [getattr(sema_enums, type_name) for type_name in sema_enums.__all__]
}

router = APIRouter()

MAX_POINTS = 10000

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

whitewire_pwr_threshold_default = 20
whitewire_pwr_threshold_overrides = {"hw1.isone.me.versant.keene.beech": 100, "hw1.isone.me.versant.keene.elm": 1}

def determine_query_channels(channels: list[str]):

    # Add the corresponding whitewire-pwr channels for any requested heatcall channels
    whitewire_pwr_channels = []
    for ch in channels:
        if 'heatcall' in ch:
            whitewire_pwr_channels.append(ch.replace('heatcall', 'whitewire-pwr'))
    channels.extend(whitewire_pwr_channels)

    def is_regex(x):
        return x[0] == '^' and x[-1] == '$'
    def is_not_regex(x):
        return not is_regex(x)
    in_channels = list(filter(is_not_regex, channels))
    like_channels = list(filter(is_regex, channels))

    return in_channels, like_channels

def post_process_channel_readings(installation_id: str, channel_readings: list[ChannelReadingsListItem]):
    readings_by_name = {x.channel_name: x for x in channel_readings}

    # Populate the heatcall channels if necessary for any whitewire-pwr channels
    whitewire_pwr_readings = filter(lambda x: 'whitewire-pwr' in x.channel_name, channel_readings)
    for r in whitewire_pwr_readings:
        heatcall_channel_name = r.channel_name.replace('whitewire-pwr', 'heatcall')
        if heatcall_channel_name not in readings_by_name:
            threshold = whitewire_pwr_threshold_overrides.get(installation_id, whitewire_pwr_threshold_default)
            heatcall_reading = ChannelReadingsListItem(
                channel_name=heatcall_channel_name,
                # TODO pull from SEMA
                unit='Unitless',
                unit_type='gw1.unit',
                value_list=[1 if x and abs(x) > threshold else 0 for x in r.value_list]
            )
            channel_readings.append(heatcall_reading)


def query_readings_with_times(db: Session, start: datetime, end: datetime, time_step_seconds: int, installation_id: str, in_channels: list[str], like_channels: list[str]):
    # To get an accurate and complete set of time-averaged data for the requested time range,
    # our query needs to include the last value from before our time range begins.
    # Otherwise, data will be missing for any of our time buckets that end before the timestamp of our first value.
    # Additionally, the first time bucket that actually does contain a value will not be able to compute an accurate 
    # average value, since it won't know its starting value.
    # 
    # We have no good way to know how far back to search, so we just go one time step and hope that it's enough.
    # Maybe we should increase this to a full hour?
    db_query_start = start - timedelta(seconds=time_step_seconds)

    # Additionally, we need to query for a full time step after our time range so that we can calculate the average value
    # of the time bucket that begins at the requested end time.
    db_query_end = end + timedelta(seconds=time_step_seconds)

    query_interval = text(f"INTERVAL '{time_step_seconds} seconds'")

    # The innermost query gets the time-weighted interval data for the selected time range, terminal asset, and channels
    # 
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
        ReadingChannelSql.terminal_asset_alias == installation_id + ".ta",
        or_(
            ReadingChannelSql.name.in_(in_channels),
            *map(lambda x: ReadingChannelSql.name.regexp_match(x), like_channels)
        )
    ).group_by(
        text('time_bucket'),
        ReadingChannelSql.name,
        ReadingChannelSql.unit,
        ReadingChannelSql.unit_type
    ).order_by(
        ReadingChannelSql.name,
        text('time_bucket')
    ).subquery()

    # The next query calculates the time-weighted average for the data
    #
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

    # The outermost query fills in gaps where there was no data
    #
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


    result_bucket_count = math.floor((end - start).total_seconds() / time_step_seconds) + 1
    db_result_bucket_count = result_bucket_count + 2

    # Our result is a list of rows of (name, time, value).
    # Each name will have db_result_bucket_count consecutive entries in ascending time order
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

    return channel_readings, times

def query_late_persistence(db: Session, start: datetime, end: datetime, installation_id: str):

    # select timestamp, is_delayed from (
    # 	select timestamp, is_delayed, is_delayed <> LAG(is_delayed) OVER (ORDER BY timestamp) as is_delay_changed
    # 	from (
    # 		select timestamp, persisted_at - created_at > '1 minute' as is_delayed
    # 		from messages
    # 		where from_alias like '%spruce%'
    # 	)
    # )
    # where (is_delay_changed IS NULL OR is_delay_changed)
    # order by timestamp

    is_delayed_query = select(
        MessageSql.timestamp.label('timestamp'),
        ((MessageSql.persisted_at - MessageSql.created_at) > text("INTERVAL '5 minutes'")).label('is_delayed')
    ).where(
        MessageSql.from_alias == installation_id + ".scada",
        MessageSql.message_type_name == 'report.event',
        MessageSql.timestamp >= start,
        MessageSql.timestamp <= end
    ).subquery()

    is_delayed_changed_query = select(
        is_delayed_query.c.timestamp,
        is_delayed_query.c.is_delayed,
        (is_delayed_query.c.is_delayed != func.lag(is_delayed_query.c.is_delayed).over(order_by=text('timestamp'))).label('is_delayed_changed')
    ).subquery()

    changelist_query = select(
        is_delayed_changed_query.c.timestamp,
        is_delayed_changed_query.c.is_delayed
    ).where(
        or_(
            is_delayed_changed_query.c.is_delayed_changed.is_(None),
            is_delayed_changed_query.c.is_delayed_changed
        )
    )

    db_result = db.execute(changelist_query).all()

    result: list[tuple[str, str]] = []
    delay_start = None
    for row in db_result:
        [timestamp, is_delayed] = row
        if is_delayed:
            delay_start = timestamp
        elif delay_start is not None:
            result.append((datetime_to_sema(delay_start), datetime_to_sema(timestamp)))
            delay_start = None

    if delay_start is not None:
        result.append((datetime_to_sema(delay_start), datetime_to_sema(end)))

    return result

def query_operating_state_sequences(db, start, end, installation_id):
    # SELECT * FROM (
    #     SELECT name, timestamp, value, value - LAG(value) OVER (PARTITION BY name ORDER BY timestamp) as diff
    #     FROM readings r
    #     JOIN reading_channels rc on rc.id = r.channel_id
    #     WHERE 
    #         rc.terminal_asset_alias like '%beech%'
    #  		  AND rc.unit='Enum'
    #         AND timestamp > '2026-04-20'
    #         AND timestamp < '2026-04-28'
    # )
    # WHERE 
    #     (diff IS NULL OR diff != 0)
    # ORDER BY timestamp;
    state_diff_query = select(
        ReadingChannelSql.name.label('name'),
        ReadingChannelSql.unit_type.label('enum_type_name'),
        ReadingSql.timestamp.label('timestamp'),
        ReadingSql.value.label('value'),
        (ReadingSql.value - func.lag(ReadingSql.value).over(partition_by=text('name'), order_by=text('timestamp'))).label('diff')
    ).join(ReadingChannelSql).where(
        ReadingChannelSql.terminal_asset_alias == installation_id + '.ta',
        ReadingChannelSql.unit == 'Enum',
        ReadingSql.timestamp >= start,
        ReadingSql.timestamp <= end,
    ).subquery()

    is_diff_query = select(
        state_diff_query.c.name,
        state_diff_query.c.enum_type_name,
        state_diff_query.c.timestamp,
        state_diff_query.c.value,
    ).where(
        or_(
            state_diff_query.c.diff.is_(None),
            state_diff_query.c.diff != 0
        )
    ).order_by(state_diff_query.c.timestamp)

    db_result = db.execute(is_diff_query).all()
    
    state_sequences: dict[str, OperatingStateSequence] = {}
    for row in db_result:
        [name, enum_type_name, timestamp, value] = row
        
        if name not in state_sequences:
            state_sequences[name] = OperatingStateSequence(
                channel_name=name,
                timestamp_list=[],
                value_list=[]
            )

        value_str = str(value)
        enum_type = SEMA_ENUM_LOOKUP.get(enum_type_name)
        if enum_type is not None:
            value_str = enum_type.values()[value]
        state_sequences[name].timestamp_list.append(datetime_to_sema(timestamp))
        state_sequences[name].value_list.append(value_str)
        
    return list(state_sequences.values())

@router.get('/api/v2/installations/{installation_id}/synced.readings.bundle')
def get_readings(installation_id, query: Annotated[ReadingsQueryParams, Query()], db: Session = Depends(get_db)):
    
    time_range_seconds = (query.end - query.start).total_seconds()
    time_step_seconds = query.time_step if query.time_step else next(i for i in DEFAULT_TIME_STEPS if i >= time_range_seconds / MAX_POINTS)

    channels = query.channels.split(',')
    in_channels, like_channels = determine_query_channels(channels)

    channel_readings, times = query_readings_with_times(db, query.start, query.end, time_step_seconds, installation_id, in_channels, like_channels)
    post_process_channel_readings(installation_id, channel_readings)

    result = SyncedReadingsBundle(
        about_g_node_alias=installation_id + ".ta",
        start_timestamp=datetime_to_sema(query.start),
        end_timestamp=datetime_to_sema(query.end),
        timestamp_list=times,
        channel_readings_list=channel_readings,
        late_persistence_list=query_late_persistence(db, query.start, query.end, installation_id),
        operating_state_sequence_list=query_operating_state_sequences(db, query.start, query.end, installation_id)
    )

    return result

# 1. All computed synthetic channels will be computed on-demand.
# 2. The computations that involve multiplication/division need to be queried separately at a 1-second interval, 
#   then computed and coalesced into the actual query interval.

    # Anytime we're doing simple addition or subtraction, the timescale is irrelevant.
    # [(a1 + b1) + (a2 + b2)] / 2 === [(a1+a2)/2 + (b1 + b2)/2]
    # If we are doing multiplication/division though:
    # [(a1*b1) + (a2*b2)] / 2 ~ [(a1+a2)/2*(b1+b2)/2 --> a1b1/4 + a1b2/4 ]


#   These can be calculated on the way out via SQL or Python
    # SyntheticChannel(
    #     name="hp-elec-in",
    #     display_name="Heat Pump Electrical Power In",
    #     unit=Gw1Unit.WattHours,
    # ),
    # SyntheticChannel(
    #     name="hp-delta-t", display_name="Heat Pump Delta-T", unit=Gw1Unit.FahrenheitX100
    # ),
    # SyntheticChannel(
    #     name="hp-heat-out",
    #     display_name="Heat Pump Thermal Power Out",
    #     unit=Gw1Unit.WattHours,
    # ),
    # SyntheticChannel(
    #     name="hp-cop", display_name="Heat Pump COP", unit=Gw1Unit.Unitless
    # ),
    # SyntheticChannel(
    #     name="dist-delta-t",
    #     display_name="Distribution Delta-T",
    #     unit=Gw1Unit.FahrenheitX100,
    # ),
    # SyntheticChannel(
    #     name="dist-heat",
    #     display_name="Distribution Thermal Power",
    #     unit=Gw1Unit.WattHours,
    # ),
    # SyntheticChannel(
    #     name="store-delta-t", display_name="Store Delta-T", unit=Gw1Unit.FahrenheitX100
    # ),
    # SyntheticChannel(
    #     name="store-flow-rate", display_name="Store Flow Rate", unit=Gw1Unit.GpmX100
    # ),
    # SyntheticChannel(
    #     name="store-heat-change",
    #     display_name="Store Thermal Power Change",
    #     unit=Gw1Unit.WattHours,
    # ),
