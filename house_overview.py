"""
Plot true house energy (heat pump true - storage in true - buffer in true) over time.
True series at CSV timestep: storage and buffer energy from CSV temps (like storage_overview);
HP heat from primary-flow, hp-lwt, hp-ewt (see add_hourly_data.py). Expected from FLO load_forecast[0].
"""

import json
import dotenv
import numpy as np
import pendulum
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch
from pathlib import Path
from sqlalchemy import create_engine, select, asc, cast, BigInteger
from sqlalchemy.orm import sessionmaker

from config import Settings
from models import MessageSql
from gridflo.asl.types import FloParamsHouse0
from gridflo import Flo
from gridflo.dijkstra_types import DEdge

# Same CSV discovery and load as storage_overview
script_dir = Path(__file__).resolve().parent
matches = sorted(script_dir.glob("beech_*s_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not matches:
    raise FileNotFoundError(f'No file matching "beech_*s_*.csv" in {script_dir}')
CSV_PATH = matches[0]
print(f"Using: {CSV_PATH.name}")

df = pd.read_csv(CSV_PATH, skiprows=[0])
df["timestamps"] = pd.to_datetime(df["timestamps"])
if df["timestamps"].dt.tz is None:
    df["timestamps"] = df["timestamps"].dt.tz_localize("America/New_York", ambiguous="infer")
else:
    df["timestamps"] = df["timestamps"].dt.tz_convert("America/New_York")

# Relay columns for charge/discharge background
def find_relay_col(needle: str) -> str:
    cols = [c for c in df.columns if needle in c.lower()]
    if not cols:
        raise ValueError(f'No column containing "{needle}" found. Columns: {list(df.columns)}')
    return cols[0]

relay3_col = find_relay_col("relay3")
relay6_col = find_relay_col("relay6")
relay9_col = find_relay_col("relay9")
# Background regions: Green = HpOffStoreDischarge, Orange = HpOnStoreOff, Red = HpOnStoreCharge
df["store_charge"] = (df[relay3_col].astype(int) == 1).astype(int)
df["store_discharge"] = ((df[relay9_col].astype(int) == 1) & (df[relay3_col].astype(int) != 1)).astype(int)
# Red: HpOnStoreCharge = relay 3 pulled, relay 6 not pulled
df["hp_on_store_charge"] = (df[relay3_col].astype(int) == 1) & (df[relay6_col].astype(int) == 0)
# Orange: HpOnStoreOff = relay 3 not pulled, relay 6 not pulled
df["hp_on_store_off"] = (df[relay3_col].astype(int) == 0) & (df[relay6_col].astype(int) == 0)
# Green: HpOffStoreDischarge = relay 9 not pulled (drawn first so red/orange show on top and no greenish blend)
df["hp_off_store_discharge"] = (df[relay9_col].astype(int) == 0)
# Green: HpOffStoreOff = relay 9 pulled, relay 6 pulled
df["hp_off_store_off"] = (df[relay9_col].astype(int) == 1) & (df[relay6_col].astype(int) == 1)
dist_flow_col = find_relay_col("dist-flow")  # needle matches column name
df["dist_flow_high"] = (df[dist_flow_col].astype(float) / 100) > 0.5

# Timeframe from CSV
ts_min = df["timestamps"].min()
ts_max = df["timestamps"].max()
start_ms = int(ts_min.value // 1_000_000)
end_ms = int(ts_max.value // 1_000_000)
house_alias = "beech"

# FLO params: expected house energy (load_forecast[0]) per message at :57 (reuse cache from storage_overview)
settings = Settings(_env_file=dotenv.find_dotenv())
FLO_CACHE_PATH = script_dir / "flo_expected_heat_cache.json"

def _ms_to_new_york(ms: int):
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert("America/New_York")

engine = create_engine(settings.db_url_no_async.get_secret_value())
Session = sessionmaker(bind=engine)
session = Session()
stmt = select(MessageSql).filter(
    MessageSql.message_type_name == "flo.params.house0",
    MessageSql.from_alias == f"hw1.isone.me.versant.keene.{house_alias}",
    MessageSql.message_persisted_ms <= cast(end_ms + 10 * 60 * 1000, BigInteger),
    MessageSql.message_persisted_ms >= cast(start_ms - 10 * 60 * 1000, BigInteger),
).order_by(asc(MessageSql.message_persisted_ms))
messages = session.execute(stmt).scalars().all()
flo_messages = [m for m in messages if pendulum.from_timestamp(m.message_persisted_ms / 1000, tz="America/New_York").minute == 57]
session.close()
engine.dispose()

def _message_ms_with_forecast():
    return [m.message_persisted_ms for m in flo_messages if FloParamsHouse0(**m.payload).load_forecast]

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

flo_times = []
flo_load_first_hour = []  # expected house energy (load_forecast[0]) per message
cached = _load_flo_cache()
if cached is not None:
    flo_times = [_ms_to_new_york(e["ms"]) for e in cached]
    flo_load_first_hour = [e["load"] for e in cached]
    print(f"FLO expected house: loaded {len(flo_times)} from cache")
else:
    for m in flo_messages:
        fp = FloParamsHouse0(**m.payload)
        if not fp.load_forecast:
            continue
        g = Flo(fp.to_bytes())
        g.solve_dijkstra()
        g.generate_recommendation(fp.to_bytes())
        flo_times.append(_ms_to_new_york(m.message_persisted_ms))
        flo_load_first_hour.append(fp.load_forecast[0])
    print(f"FLO expected house: {len(flo_times)} messages with load_forecast (no cache)")

# --- True house energy at CSV timestep (like storage_overview) ---
# Storage energy: 360 gal, avg tank temp vs 80°F (same as storage_overview)
tank_depth_cols = [f"tank{i}-depth{j}" for i in range(1, 4) for j in range(1, 4)]
tank_cols = [c for c in tank_depth_cols if c in df.columns]
if not tank_cols:
    raise ValueError(f"No tank columns found. Available: {list(df.columns)}")
df["storage_temp_avg"] = df[tank_cols].mean(axis=1) / 100
STORAGE_VOLUME_GALLONS = 360
STORAGE_MASS_KG = STORAGE_VOLUME_GALLONS * 3.785
BASE_TEMP_F = 80
DEG_F_TO_KWH_THERMAL_STORAGE = STORAGE_MASS_KG * 4.187 * (5 / 9) / 3600
df["storage_energy_kwh"] = (df["storage_temp_avg"] - BASE_TEMP_F) * DEG_F_TO_KWH_THERMAL_STORAGE

# Buffer energy: 120 gal, avg buffer temp vs 80°F
buffer_cols = [c for c in ["buffer-depth1", "buffer-depth2", "buffer-depth3"] if c in df.columns]
if not buffer_cols:
    raise ValueError(f"No buffer columns found. Available: {list(df.columns)}")
df["buffer_temp_avg"] = df[buffer_cols].mean(axis=1) / 100
BUFFER_VOLUME_GALLONS = 120
BUFFER_MASS_KG = BUFFER_VOLUME_GALLONS * 3.785
DEG_F_TO_KWH_THERMAL_BUFFER = BUFFER_MASS_KG * 4.187 * (5 / 9) / 3600
df["buffer_energy_kwh"] = (df["buffer_temp_avg"] - BASE_TEMP_F) * DEG_F_TO_KWH_THERMAL_BUFFER

# HP heat between timesteps: primary-flow (GPM×100), hp-lwt, hp-ewt (degC×1000) — add_hourly_data.py
primary_flow_col = next((c for c in df.columns if "primary-flow" in c.lower() and "hz" not in c.lower()), None)
hp_lwt_col = next((c for c in df.columns if "hp-lwt" in c.lower()), None)
hp_ewt_col = next((c for c in df.columns if "hp-ewt" in c.lower()), None)
if not all([primary_flow_col, hp_lwt_col, hp_ewt_col]):
    raise ValueError("Need primary-flow, hp-lwt, hp-ewt columns for HP heat")
df["lift_C"] = (df[hp_lwt_col].astype(float) - df[hp_ewt_col].astype(float)) / 1000
df["flow_kg_s"] = (df[primary_flow_col].astype(float) / 100) / 60 * 3.78541
df["hp_power_kW"] = df["flow_kg_s"] * 4.187 * df["lift_C"]
df["dt_s"] = df["timestamps"].diff().dt.total_seconds()
df.loc[df.index[0], "dt_s"] = 0
df["hp_heat_kwh"] = df["hp_power_kW"] * df["dt_s"] / 3600
df["storage_in_kwh"] = df["storage_energy_kwh"].diff()
df["buffer_in_kwh"] = df["buffer_energy_kwh"].diff()
df.loc[df.index[0], "storage_in_kwh"] = 0
df.loc[df.index[0], "buffer_in_kwh"] = 0
df["house_energy_interval_kwh"] = df["hp_heat_kwh"] - df["storage_in_kwh"] - df["buffer_in_kwh"]
df["cumulative_house_kwh"] = df["house_energy_interval_kwh"].cumsum()
print(f"True house energy at CSV resolution: {len(df)} rows")

# Expected house segments: one per clock hour, (start, cumulative_true) → (end, cumulative_true + expected_kwh)
segment_start_times = []
segment_end_times = []
true_house_at_start = []
expected_house_kwh = []
if flo_times and flo_load_first_hour:
    segment_start_times = [pd.Timestamp(t).ceil("h") for t in flo_times]
    segment_end_times = [t + pd.Timedelta(hours=1) for t in segment_start_times]
    expected_house_kwh = list(flo_load_first_hour)
    right = pd.DataFrame({"t": segment_start_times, "order": range(len(segment_start_times))})
    merged = pd.merge_asof(
        right.sort_values("t"),
        df[["timestamps", "cumulative_house_kwh"]].sort_values("timestamps"),
        left_on="t",
        right_on="timestamps",
        direction="backward",
    )
    merged = merged.sort_values("order")
    true_house_at_start = merged["cumulative_house_kwh"].tolist()

# Difference (live − expected) in the current hour, at every time step; expected = actual at start of each hour (so diff = 0 there)
df["hour_start"] = df["timestamps"].dt.floor("h")
# Cumulative at start of each hour (for each row, value at its hour_start)
right = df[["timestamps", "cumulative_house_kwh"]].sort_values("timestamps")
merged_hour = pd.merge_asof(
    df[["hour_start"]].sort_values("hour_start").drop_duplicates("hour_start"),
    right,
    left_on="hour_start",
    right_on="timestamps",
    direction="backward",
)
hour_to_cumulative = merged_hour.set_index("hour_start")["cumulative_house_kwh"]
df["cumulative_at_hour_start"] = df["hour_start"].map(hour_to_cumulative)
df["live_change_in_hour_so_far"] = df["cumulative_house_kwh"] - df["cumulative_at_hour_start"]
# Prorate expected over the hour so expected = actual (= 0) at start of hour
if segment_start_times and expected_house_kwh:
    hour_to_expected = pd.Series(expected_house_kwh, index=segment_start_times)
    df["expected_in_hour"] = df["hour_start"].map(hour_to_expected)
    elapsed_s = (df["timestamps"] - df["hour_start"]).dt.total_seconds()
    df["expected_so_far_in_hour"] = df["expected_in_hour"] * (elapsed_s / 3600.0)
    df["diff_live_expected_kwh"] = df["live_change_in_hour_so_far"] - df["expected_so_far_in_hour"]
else:
    df["diff_live_expected_kwh"] = np.nan

# Single plot: live − expected per hour with charge/discharge background
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

ts = df["timestamps"]
# Red: HpOnStoreCharge (relay 3 pulled, relay 6 not pulled)
red_on = df["hp_on_store_charge"].to_numpy()
i = 0
while i < len(ts) - 1:
    if not red_on[i]:
        i += 1
        continue
    t_start = ts.iloc[i]
    j = i + 1
    while j < len(ts) - 1 and red_on[j]:
        j += 1
    t_end = ts.iloc[j]
    ax.axvspan(t_start, t_end, color="red", alpha=0.12)
    i = j
# Orange: HpOnStoreOff (relay 3 not pulled, relay 6 not pulled)
orange_on = df["hp_on_store_off"].to_numpy()
i = 0
while i < len(ts) - 1:
    if not orange_on[i]:
        i += 1
        continue
    t_start = ts.iloc[i]
    j = i + 1
    while j < len(ts) - 1 and orange_on[j]:
        j += 1
    t_end = ts.iloc[j]
    ax.axvspan(t_start, t_end, color="gold", alpha=0.12)
    i = j
# Green: HpOffStoreOff (relay 9 pulled, relay 6 pulled)
green_on = df["hp_off_store_off"].to_numpy()
i = 0
while i < len(ts) - 1:
    if not green_on[i]:
        i += 1
        continue
    t_start = ts.iloc[i]
    j = i + 1
    while j < len(ts) - 1 and green_on[j]:
        j += 1
    t_end = ts.iloc[j]
    ax.axvspan(t_start, t_end, color="green", alpha=0.12)
    i = j
# Blue: dist-flow/100 > 0.5, limited to ±2 kWh band (can overlap with others)
dist_flow_high = df["dist_flow_high"].to_numpy()
i = 0
while i < len(ts) - 1:
    if not dist_flow_high[i]:
        i += 1
        continue
    t_start = ts.iloc[i]
    j = i + 1
    while j < len(ts) - 1 and dist_flow_high[j]:
        j += 1
    t_end = ts.iloc[j]
    x0 = mdates.date2num(t_start)
    x1 = mdates.date2num(t_end)
    ax.add_patch(
        Rectangle((x0, -2), x1 - x0, 4, transform=ax.transData, facecolor="gray", alpha=0.2, linewidth=0)
    )
    i = j

# One curve per hour (not linked), with circles at start and end of each hour
if segment_start_times is not None:
    for h_start in segment_start_times:
        mask = df["hour_start"] == h_start
        if not mask.any():
            continue
        block = df.loc[mask]
        ax.plot(block["timestamps"], block["diff_live_expected_kwh"], color="tab:blue", linewidth=2)
        ax.plot(
            block["timestamps"].iloc[[0, -1]],
            block["diff_live_expected_kwh"].iloc[[0, -1]],
            "o",
            color="tab:blue",
            markersize=6,
        )
ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
ax.text(0.5, 0.92, "Less heat to house than expected", transform=ax.transAxes, fontsize=9, ha="center", va="top", color="gray")
ax.text(0.5, 0.08, "More heat to house than expected", transform=ax.transAxes, fontsize=9, ha="center", va="bottom", color="gray")
legend_handles = [
    Patch(facecolor="gold", alpha=0.12, label="HpOnStoreOff"),
    Patch(facecolor="red", alpha=0.12, label="HpOnStoreCharge"),
    Patch(facecolor="green", alpha=0.12, label="HpOffStoreOff"),
]
ax.legend(handles=legend_handles, loc="best")
ax.set_ylabel("Live − expected (kWh)")
ax.set_xlabel("Time")
ax.set_title("Difference in the current hour (live − prorated expected; 0 at start of each hour)")
ax.grid(True, alpha=0.3)

import pytz
ny_tz = pytz.timezone("America/New_York")
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=ny_tz))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=ny_tz))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
