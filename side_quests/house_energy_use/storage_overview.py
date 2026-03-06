"""
Compute average_storage_temperature from tank depth columns and apply (x-32)*5/9*10.
Reads 1-second CSV with timestamps and tank{i}-depth{j} columns (i,j in [1,2,3]).
"""

import json
import dotenv
import numpy as np
import pendulum
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sqlalchemy import asc, cast, create_engine, select, BigInteger
from sqlalchemy.orm import sessionmaker

from config import Settings
from models import MessageSql
from gridflo.asl.types import FloParamsHouse0
from gridflo import Flo
from gridflo.dijkstra_types import DEdge

# Find CSV matching beech_{i}s_*.csv (e.g. beech_1s_, beech_2s_, ...)
script_dir = Path(__file__).resolve().parent
matches = sorted(script_dir.glob("oak_*s_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not matches:
    raise FileNotFoundError(f'No file matching "beech_*s_*.csv" in {script_dir}')
CSV_PATH = matches[0]
print(f"Using: {CSV_PATH.name}")

# First row is a filename, not the header
df = pd.read_csv(CSV_PATH, skiprows=[0])
df["timestamps"] = pd.to_datetime(df["timestamps"])
# Ensure all timestamps are in New York (CSV is typically local; keep plots aligned)
if df["timestamps"].dt.tz is None:
    df["timestamps"] = df["timestamps"].dt.tz_localize("America/New_York", ambiguous="infer")
else:
    df["timestamps"] = df["timestamps"].dt.tz_convert("America/New_York")

# Build list of tank{i}-depth{j} columns for i,j in [1,2,3]
tank_depth_cols = [
    f"tank{i}-depth{j}"
    for i in range(1, 4)
    for j in range(1, 4)
]

# Keep only columns that exist in the dataframe
available_cols = [c for c in tank_depth_cols if c in df.columns]
if not available_cols:
    raise ValueError(
        f"None of the tank depth columns {tank_depth_cols} found in CSV. "
        f"Available columns: {list(df.columns)}"
    )

# Average of all tank-depth columns
df["average_storage_temperature"] = df[available_cols].mean(axis=1)

# Apply (x-32)*5/9*10
df["average_storage_temperature"] = df["average_storage_temperature"] / 100

# Convert average storage temperature to energy (kWh) vs base 80°F (same as flo_report)
STORAGE_VOLUME_GALLONS = 360
STORAGE_MASS_KG = STORAGE_VOLUME_GALLONS * 3.785
BASE_TEMP_F = 80
# kWh per deg F: mass_kg * c_water * (1 deg F in K = 5/9) / 3600
DEG_F_TO_KWH_THERMAL_STORAGE = STORAGE_MASS_KG * 4.187 * (5 / 9) / 3600
df["storage_energy_kwh"] = (df["average_storage_temperature"] - BASE_TEMP_F) * DEG_F_TO_KWH_THERMAL_STORAGE

# Optional: write result (uncomment to save)
# df.to_csv("beech_1s_with_avg_storage_temp.csv", index=False)

# Find relay3 and relay9 columns
def find_relay_col(needle: str) -> str:
    cols = [c for c in df.columns if needle in c.lower()]
    if not cols:
        raise ValueError(f'No column containing "{needle}" found. Columns: {list(df.columns)}')
    return cols[0]

relay3_col = find_relay_col("relay3")
relay9_col = find_relay_col("relay9")
relay3_pulled = (df[relay3_col].astype(int) == 1)
relay9_pulled = (df[relay9_col].astype(int) == 1)

# System state: StoreCharge (relay3), StoreDischarge (relay9), else Store Idle. If both pulled, show Charge.
df["store_charge"] = relay3_pulled.astype(int)
df["store_discharge"] = (relay9_pulled & ~relay3_pulled).astype(int)

# Crop to first 6 hours of data
# ts_min = df["timestamps"].min()
# crop_end = ts_min + pd.Timedelta(hours=6)
# df = df[df["timestamps"] < crop_end].copy()
# print(f"Cropped to first 6 hours: {ts_min} to {crop_end} ({len(df)} rows)")

# Timeframe from CSV for FLO params query
ts_min = df["timestamps"].min()
ts_max = df["timestamps"].max()
start_ms = int(ts_min.value // 1_000_000)  # nanoseconds to ms
end_ms = int(ts_max.value // 1_000_000)
house_alias = "oak"

# Query flo.params.house0 messages in CSV timeframe
settings = Settings(_env_file=dotenv.find_dotenv())
engine = create_engine(settings.db_url_no_async.get_secret_value())
Session = sessionmaker(bind=engine)
session = Session()
stmt = select(MessageSql).filter(
    MessageSql.message_type_name == "flo.params.house0",
    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}",
    MessageSql.message_persisted_ms <= cast(end_ms + 10 * 60 * 1000, BigInteger),
    MessageSql.message_persisted_ms >= cast(start_ms - 10 * 60 * 1000, BigInteger),
).order_by(asc(MessageSql.message_persisted_ms))
result = session.execute(stmt)
messages = result.scalars().all()
# Shortlist: only messages at minute 57 (same as flo_report)
flo_messages = []
for m in messages:
    if pendulum.from_timestamp(m.message_persisted_ms / 1000, tz="America/New_York").minute == 57:
        flo_messages.append(m)
print(f"Found {len(messages)} flo.params.house0 messages, {len(flo_messages)} at minute 57")
session.close()
engine.dispose()

# Cache file for expected heat from HP (avoid running Flo every time)
FLO_CACHE_PATH = script_dir / "flo_expected_heat_cache.json"

def _message_ms_with_forecast():
    """Message persisted ms for messages that have load_forecast (in order)."""
    out = []
    for m in flo_messages:
        fp = FloParamsHouse0(**m.payload)
        if fp.load_forecast:
            out.append(m.message_persisted_ms)
    return out

def _load_flo_cache():
    if not FLO_CACHE_PATH.exists():
        return None
    try:
        with open(FLO_CACHE_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("start_ms") != start_ms or data.get("end_ms") != end_ms or data.get("csv") != CSV_PATH.name:
        return None
    expected_ms = _message_ms_with_forecast()
    entries = data.get("entries") or []
    if len(entries) != len(expected_ms) or [e["ms"] for e in entries] != expected_ms:
        return None
    return entries

def _save_flo_cache(entries):
    with open(FLO_CACHE_PATH, "w") as f:
        json.dump(
            {"start_ms": start_ms, "end_ms": end_ms, "csv": CSV_PATH.name, "entries": entries},
            f,
            indent=2,
        )

# Expected load and expected heat from HP (first hour) per FLO message
flo_times = []
flo_load_first_hour = []
flo_hp_heat_minus_load = []  # expected heat from HP - load for that hour

def _ms_to_new_york(ms: int):
    """Convert epoch milliseconds (UTC) to New York time for plotting."""
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert("America/New_York")

cached = _load_flo_cache()
if cached is not None:
    flo_times = [_ms_to_new_york(e["ms"]) for e in cached]
    flo_load_first_hour = [e["load"] for e in cached]
    flo_hp_heat_minus_load = [e["hp_minus_load"] for e in cached]
    print(f"FLO params: loaded {len(flo_times)} expected-heat values from cache {FLO_CACHE_PATH.name}")
else:
    num_flos = len(flo_messages)
    for i, m in enumerate(flo_messages, start=1):
        fp = FloParamsHouse0(**m.payload)
        if not fp.load_forecast:
            continue
        print(f"\nRunning FLO {i}/{num_flos}")
        g = Flo(fp.to_bytes())
        g.solve_dijkstra()
        g.generate_recommendation(fp.to_bytes())
        initial_node_edge: DEdge = [e for e in g.bid_edges[g.initial_node] if e.head == g.initial_node.next_node][0]
        hp_heat_out_expected = initial_node_edge.hp_heat_out
        load_first = fp.load_forecast[0]
        flo_times.append(_ms_to_new_york(m.message_persisted_ms))
        flo_load_first_hour.append(load_first)
        flo_hp_heat_minus_load.append(hp_heat_out_expected - load_first)
    print(f"FLO params: {len(flo_messages)} messages, {len(flo_times)} with load_forecast and HP expected")
    if flo_times:
        _save_flo_cache(
            [
                {"ms": int(t.timestamp() * 1000), "load": load, "hp_minus_load": hml}
                for t, load, hml in zip(flo_times, flo_load_first_hour, flo_hp_heat_minus_load)
            ]
        )
        print(f"Saved expected heat cache to {FLO_CACHE_PATH.name}")

# Expected store energy: one segment per clock hour [X:00, X+1:00] so top and middle plots align.
# Segment j uses: start = top of hour after message j (0:57 → 1:00), end = start+1h; value = bottom-plot value at message j (active from 0:57 to 1:57, so covers start of this clock hour).
segment_start_times = []
segment_end_times = []
true_energy_at_start = []
if flo_times and flo_hp_heat_minus_load:
    # Clock-hour boundaries: segment j spans [ceil(flo_times[j]), ceil(flo_times[j]) + 1h], e.g. 1:00–2:00 for message at 0:57
    segment_start_times = [pd.Timestamp(t).ceil("h") for t in flo_times]
    segment_end_times = [t + pd.Timedelta(hours=1) for t in segment_start_times]
    # True storage energy at each segment start (clock hour), same timeline as middle plot
    right = pd.DataFrame({"t": segment_start_times, "order": range(len(segment_start_times))})
    merged = pd.merge_asof(
        right.sort_values("t"),
        df[["timestamps", "storage_energy_kwh"]].sort_values("timestamps"),
        left_on="t",
        right_on="timestamps",
        direction="backward",
    )
    merged = merged.sort_values("order")
    true_energy_at_start = merged["storage_energy_kwh"].tolist()

# Single plot: storage energy with charge/discharge as light background
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Light background: red when charging, green when discharging (merge contiguous intervals to avoid stripes)
ts = df["timestamps"]
charge = df["store_charge"].to_numpy()
discharge = df["store_discharge"].to_numpy()
# State from t[i] to t[i+1]: 1=charge, -1=discharge, 0=idle (charge takes precedence)
state = np.where(charge, 1, np.where(discharge, -1, 0))
i = 0
while i < len(ts) - 1:
    t_start = ts.iloc[i]
    s = state[i]
    j = i + 1
    while j < len(ts) - 1 and state[j] == s:
        j += 1
    t_end = ts.iloc[j]
    if s == 1:
        ax.axvspan(t_start, t_end, color="red", alpha=0.15)
    elif s == -1:
        ax.axvspan(t_start, t_end, color="green", alpha=0.15)
    i = j

ax.plot(df["timestamps"], df["storage_energy_kwh"], color="steelblue", label="True storage energy")
if flo_times and flo_hp_heat_minus_load and segment_start_times and true_energy_at_start:
    for i in range(len(flo_times)):
        label = "Expected storage energy" if i == 0 else None
        ax.plot(
            [segment_start_times[i], segment_end_times[i]],
            [true_energy_at_start[i], true_energy_at_start[i] + flo_hp_heat_minus_load[i]],
            color="tab:orange",
            marker=".",
            markersize=6,
            label=label,
        )
ax.set_ylabel("Storage energy (kWh)")
ax.set_xlabel("Time")
ax.set_title("Storage energy over time (vs 80°F base)")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
