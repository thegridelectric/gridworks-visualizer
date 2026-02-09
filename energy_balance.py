"""
Single plot with two lines over time:
  - HP actual − buffer actual − house expected  (expected storage in)
  - Storage actual energy (kWh)
Same CSV, relay logic, and FLO cache as house_overview.py.
"""

import json
import dotenv
import numpy as np
import pendulum
import pytz
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

# CSV discovery and load
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

# Relay columns
def find_col(needle: str) -> str:
    cols = [c for c in df.columns if needle in c.lower()]
    if not cols:
        raise ValueError(f'No column containing "{needle}" found. Columns: {list(df.columns)}')
    return cols[0]

relay3_col = find_col("relay3")
relay6_col = find_col("relay6")
relay9_col = find_col("relay9")
# Background states
df["hp_on_store_charge"] = (df[relay3_col].astype(int) == 1) & (df[relay6_col].astype(int) == 0)
df["hp_on_store_off"] = (df[relay3_col].astype(int) == 0) & (df[relay6_col].astype(int) == 0)
df["hp_off_store_off"] = (df[relay9_col].astype(int) == 1) & (df[relay6_col].astype(int) == 1)
dist_flow_col = find_col("dist-flow")
df["dist_flow_high"] = (df[dist_flow_col].astype(float) / 100) > 0.5
df["dist_flow_on"] = (df[dist_flow_col].astype(float) / 100) > 0

# --- Storage energy (360 gal, avg tank temp vs 80°F) ---
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

# --- Buffer energy (120 gal, avg buffer temp vs 80°F) ---
buffer_cols = [c for c in ["buffer-depth1", "buffer-depth2", "buffer-depth3"] if c in df.columns]
if not buffer_cols:
    raise ValueError(f"No buffer columns found. Available: {list(df.columns)}")
df["buffer_temp_avg"] = df[buffer_cols].mean(axis=1) / 100
BUFFER_VOLUME_GALLONS = 120
BUFFER_MASS_KG = BUFFER_VOLUME_GALLONS * 3.785
DEG_F_TO_KWH_THERMAL_BUFFER = BUFFER_MASS_KG * 4.187 * (5 / 9) / 3600
df["buffer_energy_kwh"] = (df["buffer_temp_avg"] - BASE_TEMP_F) * DEG_F_TO_KWH_THERMAL_BUFFER

# --- HP heat per interval ---
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

# --- Cumulative quantities ---
df["cumulative_hp_kwh"] = df["hp_heat_kwh"].cumsum()
df["buffer_in_kwh"] = df["buffer_energy_kwh"].diff()
df.loc[df.index[0], "buffer_in_kwh"] = 0
df["cumulative_buffer_in_kwh"] = df["buffer_in_kwh"].cumsum()

# --- FLO expected house energy (load_forecast[0]) per hour ---
ts_min = df["timestamps"].min()
ts_max = df["timestamps"].max()
start_ms = int(ts_min.value // 1_000_000)
end_ms = int(ts_max.value // 1_000_000)
house_alias = "beech"

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
flo_load_first_hour = []
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

# Fallback: if no FLO messages, use expected_house_kwh from the hourly electricity CSV
if not flo_times or not flo_load_first_hour:
    elec_matches = sorted(script_dir.glob("beech_electricity_use_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if elec_matches:
        elec_path = elec_matches[0]
        edf = pd.read_csv(elec_path)
        if "expected_house_kwh" in edf.columns:
            edf["hour_start"] = pd.to_datetime(edf["hour_start"])
            if edf["hour_start"].dt.tz is None:
                edf["hour_start"] = edf["hour_start"].dt.tz_localize("America/New_York", ambiguous="infer")
            else:
                edf["hour_start"] = edf["hour_start"].dt.tz_convert("America/New_York")
            edf = edf.sort_values("hour_start").reset_index(drop=True)
            flo_times = list(edf["hour_start"])
            flo_load_first_hour = list(edf["expected_house_kwh"])
            print(f"Using expected_house_kwh from {elec_path.name}: {len(flo_times)} hours")

# Build cumulative expected house energy at every CSV timestep
df["hour_start"] = df["timestamps"].dt.floor("h")
segment_start_times = []
if flo_times and flo_load_first_hour:
    # When from FLO: ceil to next hour. When from hourly CSV: hour_start is already the hour.
    # Use the hour_start directly if it's on the hour, otherwise ceil.
    segment_start_times = [pd.Timestamp(t).ceil("h") if pd.Timestamp(t).minute != 0 else pd.Timestamp(t) for t in flo_times]
    hour_to_expected = pd.Series(flo_load_first_hour, index=segment_start_times)
    df["expected_in_hour"] = df["hour_start"].map(hour_to_expected)
    # Elapsed fraction within the hour
    elapsed_s = (df["timestamps"] - df["hour_start"]).dt.total_seconds()
    df["expected_so_far_in_hour"] = df["expected_in_hour"] * (elapsed_s / 3600.0)
    # Cumulative expected: sum of completed hours + prorated current hour
    unique_hours = sorted(df["hour_start"].dropna().unique())
    hour_expected_full = {h: hour_to_expected.get(h, np.nan) for h in unique_hours}
    cumsum_at_hour_start = {}
    running = 0.0
    for h in unique_hours:
        cumsum_at_hour_start[h] = running
        val = hour_expected_full[h]
        if not np.isnan(val):
            running += val
    df["cumulative_expected_house_at_hour_start"] = df["hour_start"].map(cumsum_at_hour_start)
    df["cumulative_expected_house_kwh"] = df["cumulative_expected_house_at_hour_start"] + df["expected_so_far_in_hour"].fillna(0)
else:
    df["cumulative_expected_house_kwh"] = np.nan

# Per-hour expected storage trajectory: starts at actual storage, then diverges based on
# HP actual change + buffer actual change − house expected change within the hour
if flo_times and flo_load_first_hour:
    # Get actual values at each hour start
    hour_starts_df = df.groupby("hour_start").first()[["storage_energy_kwh", "cumulative_hp_kwh", "cumulative_buffer_in_kwh"]].rename(
        columns={"storage_energy_kwh": "storage_at_h", "cumulative_hp_kwh": "hp_at_h", "cumulative_buffer_in_kwh": "buf_at_h"}
    )
    df = df.merge(hour_starts_df, on="hour_start", how="left")
    df["expected_storage_trajectory"] = (
        df["storage_at_h"]
        + (df["cumulative_hp_kwh"] - df["hp_at_h"])
        - (df["cumulative_buffer_in_kwh"] - df["buf_at_h"])
        - df["expected_so_far_in_hour"].fillna(0)
    )

print(f"Loaded {len(df)} rows")

# ---- Plotting ----
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[5, 1])
ny_tz = pytz.timezone("America/New_York")
ts = df["timestamps"]

# Background colors on bottom strip
def draw_backgrounds(target_ax):
    """Draw relay-based background colors on an axis."""
    # Red: HpOnStoreCharge
    arr = df["hp_on_store_charge"].to_numpy()
    i = 0
    while i < len(ts) - 1:
        if not arr[i]:
            i += 1
            continue
        t_start = ts.iloc[i]
        j = i + 1
        while j < len(ts) - 1 and arr[j]:
            j += 1
        target_ax.axvspan(t_start, ts.iloc[j], color="red", alpha=0.12)
        i = j
    # Gold: HpOnStoreOff
    arr = df["hp_on_store_off"].to_numpy()
    i = 0
    while i < len(ts) - 1:
        if not arr[i]:
            i += 1
            continue
        t_start = ts.iloc[i]
        j = i + 1
        while j < len(ts) - 1 and arr[j]:
            j += 1
        target_ax.axvspan(t_start, ts.iloc[j], color="gold", alpha=0.12)
        i = j
    # Green: HpOffStoreOff
    arr = df["hp_off_store_off"].to_numpy()
    i = 0
    while i < len(ts) - 1:
        if not arr[i]:
            i += 1
            continue
        t_start = ts.iloc[i]
        j = i + 1
        while j < len(ts) - 1 and arr[j]:
            j += 1
        target_ax.axvspan(t_start, ts.iloc[j], color="green", alpha=0.12)
        i = j

draw_backgrounds(ax2)

# Heat calls: gray band where dist-flow > 0 (on bottom strip)
dist_flow_on = df["dist_flow_on"].to_numpy()
i = 0
while i < len(ts) - 1:
    if not dist_flow_on[i]:
        i += 1
        continue
    t_start = ts.iloc[i]
    j = i + 1
    while j < len(ts) - 1 and dist_flow_on[j]:
        j += 1
    t_end = ts.iloc[j]
    ax2.axvspan(t_start, t_end, color="gray", alpha=0.15)
    i = j

ax2.set_yticks([])
ax2.set_ylabel("")
ax2.grid(False)

# --- Live store_change from flow integration (same method as add_hourly_data.py) ---
# store_lift_C: sign depends on relay3 (charge vs discharge)
#   relay3==0 (discharge): (store-hot-pipe - store-cold-pipe) / 1000
#   relay3==1 (charge): (store-cold-pipe - store-hot-pipe) / 1000
# store_flow: relay3==0 uses store-flow, relay3==1 uses primary-flow (both GPM×100)
store_hot_col = find_col("store-hot-pipe")
store_cold_col = find_col("store-cold-pipe")
store_flow_col = find_col("store-flow")
r3 = df[relay3_col].astype(int)
df["store_lift_C"] = np.where(
    r3 == 0,
    (df[store_hot_col].astype(float) - df[store_cold_col].astype(float)) / 1000,
    (df[store_cold_col].astype(float) - df[store_hot_col].astype(float)) / 1000,
)
df["store_flow_kgs"] = np.where(
    r3 == 0,
    df[store_flow_col].astype(float) / 100 / 60 * 3.78541,
    df[primary_flow_col].astype(float) / 100 / 60 * 3.78541,
)
df["store_heat_power_kW"] = df["store_flow_kgs"] * 4187 * df["store_lift_C"] / 1000
df["store_heat_kwh"] = df["store_heat_power_kW"] * df["dt_s"] / 3600
# store_change = negative of heat output (heat leaving storage = storage loses energy)
df["store_change_cumulative"] = -df["store_heat_kwh"].cumsum()
# Live storage from flow: starts at actual storage energy at t=0
initial_storage = df["storage_energy_kwh"].iloc[0]
df["storage_from_flow"] = initial_storage + df["store_change_cumulative"]

# Storage from flow integration: per-hour segments starting at actual storage
for idx, h_start in enumerate(sorted(df["hour_start"].unique())):
    mask = df["hour_start"] == h_start
    if not mask.any():
        continue
    block = df.loc[mask]
    s_at_start = block["storage_energy_kwh"].iloc[0]
    # Cumulative heat output within the hour, starting at 0
    block_heat = block["store_heat_kwh"].values.copy()
    block_heat[0] = 0  # no change at the start of the hour
    trajectory = s_at_start - np.cumsum(block_heat)
    label = "S_a flow" if idx == 0 else None
    ax.plot(block["timestamps"], trajectory, color="tab:green", linewidth=1.5, label=label)
    ax.plot(
        [block["timestamps"].iloc[0], block["timestamps"].iloc[-1]],
        [trajectory[0], trajectory[-1]],
        "o", color="tab:green", markersize=4,
    )

# Storage actual (continuous)
ax.plot(ts, df["storage_energy_kwh"], color="tab:blue", linewidth=1.5, label="S_a")

# Expected storage trajectory: one segment per hour, starting at actual storage
if flo_times and flo_load_first_hour:
    for idx, h_start in enumerate(segment_start_times):
        mask = df["hour_start"] == h_start
        if not mask.any():
            continue
        block = df.loc[mask]
        label = "HP_a - B_a - H_e" if idx == 0 else None
        ax.plot(block["timestamps"], block["expected_storage_trajectory"], color="tab:red", linewidth=1.5, label=label)
        ax.plot(
            block["timestamps"].iloc[[0, -1]],
            block["expected_storage_trajectory"].iloc[[0, -1]],
            "o", color="tab:red", markersize=4,
        )

ax.legend(loc="best")
ax.set_ylabel("Energy (kWh)")
ax.set_title("HP_a - B_a - H_e compared to S_a")
ax.grid(True, axis="x", alpha=0.3)

ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=ny_tz))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=ny_tz))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
