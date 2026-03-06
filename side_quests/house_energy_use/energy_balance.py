"""
Single plot with two lines over time:
  - HP actual − buffer actual − house expected  (expected storage in)
  - Storage actual energy (kWh)
Same CSV, relay logic, and FLO cache as house_overview.py.
"""

HOUSE_ALIAS = "elm"

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
data_dir = script_dir / "data"
matches = sorted(data_dir.glob(f"{HOUSE_ALIAS}_*s_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not matches:
    raise FileNotFoundError(f'No file matching "{HOUSE_ALIAS}_*s_*.csv" in {data_dir}')
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
df["hp_off_store_discharge"] = (df[relay9_col].astype(int) == 0)  # relay 9 not pulled
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

# --- Distribution energy from flow integration (dist-flow × (dist-swt - dist-rwt)) ---
dist_swt_col = find_col("dist-swt")
dist_rwt_col = find_col("dist-rwt")
df["dist_lift_C"] = (df[dist_swt_col].astype(float) - df[dist_rwt_col].astype(float)) / 1000
df["dist_flow_kgs"] = df[dist_flow_col].astype(float) / 100 / 60 * 3.78541
df["dist_heat_power_kW"] = df["dist_flow_kgs"] * 4187 * df["dist_lift_C"] / 1000
df["dist_heat_kwh"] = df["dist_heat_power_kW"] * df["dt_s"] / 3600
df["cumulative_dist_kwh"] = df["dist_heat_kwh"].cumsum()

# --- Buffer energy change from flow integration (mode-dependent flow × buffer deltaT) ---
try:
    buffer_hot_col = find_col("buffer-hot-pipe")
    buffer_cold_col = find_col("buffer-cold-pipe")
    store_flow_col_early = find_col("store-flow")
    df["buffer_lift_C"] = (df[buffer_hot_col].astype(float) - df[buffer_cold_col].astype(float)) / 1000
    pf = df[primary_flow_col].astype(float)
    sf = df[store_flow_col_early].astype(float)
    distf = df[dist_flow_col].astype(float)
    conditions = [
        df["hp_off_store_off"],
        df["hp_on_store_off"],
    ]
    choices = [
        sf - distf,
        pf - distf,
    ]
    df["buffer_flow_gpm100"] = np.select(conditions, choices, default=-distf)
    df["buffer_flow_kgs"] = df["buffer_flow_gpm100"] / 100 / 60 * 3.78541
    df["buffer_heat_power_kW"] = df["buffer_flow_kgs"] * 4187 * df["buffer_lift_C"] / 1000
    df["buffer_heat_kwh_flow"] = df["buffer_heat_power_kW"] * df["dt_s"] / 3600
    df["cumulative_buffer_flow_kwh"] = df["buffer_heat_kwh_flow"].cumsum()
except Exception as e:
    print(f"Buffer flow integration skipped: {e}")
    df["cumulative_buffer_flow_kwh"] = 0.0

# --- Cumulative quantities ---
df["cumulative_hp_kwh"] = df["hp_heat_kwh"].cumsum()
df["buffer_in_kwh"] = df["buffer_energy_kwh"].diff()
df.loc[df.index[0], "buffer_in_kwh"] = 0
df["cumulative_buffer_in_kwh"] = df["buffer_in_kwh"].cumsum()

# --- Live store_change from flow integration (same method as add_hourly_data.py) ---
try:
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
    df["store_change_cumulative"] = -df["store_heat_kwh"].cumsum()
    initial_storage = df["storage_energy_kwh"].iloc[0]
    df["storage_from_flow"] = initial_storage + df["store_change_cumulative"]
except Exception as e:
    print(f"Store flow integration skipped: {e}")
    df["storage_from_flow"] = df["storage_energy_kwh"]

# --- FLO expected house energy (load_forecast[0]) per hour ---
ts_min = df["timestamps"].min()
ts_max = df["timestamps"].max()
start_ms = int(ts_min.value // 1_000_000)
end_ms = int(ts_max.value // 1_000_000)
house_alias = HOUSE_ALIAS

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
    elec_matches = sorted(data_dir.glob(f"{HOUSE_ALIAS}_electricity_use_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
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

# ---------------------------------------------------------------------------
# Fit ALPHA, BETA, GAMMA, DELTA so that
#   H_e2 = ALPHA + BETA*oat_f + GAMMA*ws_mph*(65-oat_f) + DELTA*solar_w_m2
# best predicts the actual end-of-hour D_a (distribution energy).
# Solar irradiance (W/m2) comes from the weather station CSV.
# ---------------------------------------------------------------------------

WEATHER_CSV = "weather_KMEMILLI18.csv_processed.csv"

def _load_weather_solar(script_dir):
    """Load weather CSV and return a dict mapping hour_start -> Solar_W/m2."""
    weather_path = data_dir / WEATHER_CSV
    if not weather_path.exists():
        print(f"Weather CSV not found: {weather_path}")
        return {}
    wdf = pd.read_csv(weather_path)
    wdf["hour_start"] = pd.to_datetime(wdf["Datetime"])
    if wdf["hour_start"].dt.tz is None:
        wdf["hour_start"] = wdf["hour_start"].dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        wdf["hour_start"] = wdf["hour_start"].dt.tz_convert("America/New_York")
    solar_map = dict(zip(wdf["hour_start"], wdf["Solar_W/m2"]))
    print(f"Weather solar: loaded {len(solar_map)} hours from {WEATHER_CSV}")
    # Sanity-check OAT and wind speed
    oat_map = dict(zip(wdf["hour_start"], wdf["Temperature_F"]))
    ws_map = dict(zip(wdf["hour_start"], wdf["Wind_Speed_Mph"]))
    return solar_map, oat_map, ws_map

_solar_map, _weather_oat, _weather_ws = _load_weather_solar(script_dir) or ({}, {}, {})

def fit_he2_coefficients(df, segment_start_times, hourly_csv_path, solar_map):
    """Return (ALPHA, BETA, GAMMA, DELTA) fitted via least-squares, and the
    hourly DataFrame with oat_f / ws_mph / solar / he2 columns.
    Returns (None, None, None, None, None) if fitting is not possible."""
    if hourly_csv_path is None:
        return None, None, None, None, None
    edf = pd.read_csv(hourly_csv_path)
    if "oat_f" not in edf.columns or "ws_mph" not in edf.columns:
        return None, None, None, None, None
    edf["hour_start"] = pd.to_datetime(edf["hour_start"])
    if edf["hour_start"].dt.tz is None:
        edf["hour_start"] = edf["hour_start"].dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        edf["hour_start"] = edf["hour_start"].dt.tz_convert("America/New_York")

    # Map solar irradiance from weather CSV
    edf["solar"] = edf["hour_start"].map(solar_map).fillna(0)

    # Sanity-check: compare OAT and wind speed
    if _weather_oat:
        edf["weather_oat"] = edf["hour_start"].map(_weather_oat)
        matched = edf.dropna(subset=["weather_oat"])
        if len(matched) > 0:
            oat_diff = (matched["oat_f"] - matched["weather_oat"]).abs().mean()
            print(f"  OAT sanity check: mean |elec - weather| = {oat_diff:.2f} °F")
    if _weather_ws:
        edf["weather_ws"] = edf["hour_start"].map(_weather_ws)
        matched = edf.dropna(subset=["weather_ws"])
        if len(matched) > 0:
            ws_diff = (matched["ws_mph"] - matched["weather_ws"]).abs().mean()
            print(f"  Wind sanity check: mean |elec - weather| = {ws_diff:.2f} mph")

    # Compute end-of-hour D_a for every available hour
    da_end = {}
    for h_start in segment_start_times:
        mask = df["hour_start"] == h_start
        if not mask.any():
            continue
        block = df.loc[mask]
        da_end[h_start] = (block["cumulative_dist_kwh"].iloc[-1]
                           - block["cumulative_dist_kwh"].iloc[0])

    # Match with weather data
    rows = []
    for _, row in edf.iterrows():
        hs = row["hour_start"]
        if hs in da_end:
            rows.append({"oat_f": row["oat_f"], "ws_mph": row["ws_mph"],
                         "solar": row["solar"], "da": da_end[hs]})
    if len(rows) < 4:
        print(f"H_e2 fit: not enough matched hours ({len(rows)})")
        return None, None, None, None, None

    fit_df = pd.DataFrame(rows)
    # Design matrix:  da = ALPHA + BETA*oat_f + GAMMA*ws_mph*(65-oat_f) + DELTA*solar_w_m2
    X = np.column_stack([
        np.ones(len(fit_df)),
        fit_df["oat_f"].values,
        fit_df["ws_mph"].values * (65 - fit_df["oat_f"].values),
        fit_df["solar"].values,
    ])
    y = fit_df["da"].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta, gamma, delta = coeffs
    print(f"H_e2 fit: ALPHA={alpha:.4f}, BETA={beta:.4f}, GAMMA={gamma:.6f}, DELTA={delta:.6f}  (from {len(rows)} hours)")

    edf["he2"] = (alpha + beta * edf["oat_f"]
                  + gamma * edf["ws_mph"] * (65 - edf["oat_f"])
                  + delta * edf["solar"])
    return alpha, beta, gamma, delta, edf

elec_matches_he2 = sorted(data_dir.glob(f"{HOUSE_ALIAS}_electricity_use_*.csv"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
_hourly_path_he2 = elec_matches_he2[0] if elec_matches_he2 else None

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

# Fit H_e2 coefficients from actual D_a end-of-hour values
ALPHA, BETA, GAMMA, DELTA, _edf_he2 = fit_he2_coefficients(df, segment_start_times, _hourly_path_he2, _solar_map)
he2_per_hour = {}
if _edf_he2 is not None:
    he2_per_hour = dict(zip(_edf_he2["hour_start"], _edf_he2["he2"]))

# Per-hour expected storage trajectory: starts at actual storage, then diverges based on
# HP actual change + buffer actual change − house expected change within the hour
if flo_times and flo_load_first_hour:
    # Get values at each hour start — use storage_from_flow instead of storage_energy_kwh
    hour_starts_df = df.groupby("hour_start").first()[["storage_from_flow", "cumulative_hp_kwh", "cumulative_buffer_in_kwh"]].rename(
        columns={"storage_from_flow": "storage_flow_at_h", "cumulative_hp_kwh": "hp_at_h", "cumulative_buffer_in_kwh": "buf_at_h"}
    )
    df = df.merge(hour_starts_df, on="hour_start", how="left")
    df["expected_storage_trajectory"] = (
        df["storage_flow_at_h"]
        + (df["cumulative_hp_kwh"] - df["hp_at_h"])
        - (df["cumulative_buffer_in_kwh"] - df["buf_at_h"])
        - df["expected_so_far_in_hour"].fillna(0)
    )

    # Same but with buffer from flow integration instead of temperature
    hour_starts_flow = df.groupby("hour_start").first()[["cumulative_buffer_flow_kwh"]].rename(
        columns={"cumulative_buffer_flow_kwh": "buf_flow_at_h"}
    )
    df = df.merge(hour_starts_flow, on="hour_start", how="left", suffixes=("", "_dup"))
    df["expected_storage_trajectory_buf_flow"] = (
        df["storage_flow_at_h"]
        + (df["cumulative_hp_kwh"] - df["hp_at_h"])
        - (df["cumulative_buffer_flow_kwh"] - df["buf_flow_at_h"])
        - df["expected_so_far_in_hour"].fillna(0)
    )

    # Same trajectory but using H_e2 (with solar gains) instead of H_e
    df["he2_in_hour"] = df["hour_start"].map(he2_per_hour)
    elapsed_s = (df["timestamps"] - df["hour_start"]).dt.total_seconds()
    df["he2_so_far_in_hour"] = df["he2_in_hour"] * (elapsed_s / 3600.0)
    df["expected_storage_trajectory_he2"] = (
        df["storage_flow_at_h"]
        + (df["cumulative_hp_kwh"] - df["hp_at_h"])
        - (df["cumulative_buffer_flow_kwh"] - df["buf_flow_at_h"])
        - df["he2_so_far_in_hour"].fillna(0)
    )

print(f"Loaded {len(df)} rows")

# ---- Plotting ----
fig, (ax, ax_detail, ax2) = plt.subplots(3, 1, figsize=(12, 7), sharex=True, height_ratios=[3, 5, 1])
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

# # --- Buffer energy plot: temperature (continuous) + flow (per-hour segments) ---
# # Continuous: buffer energy from average temperature
# ax_buf.plot(ts, df["buffer_energy_kwh"], color="tab:blue", linewidth=1.5, label="B_a (temp)")
# # Per-hour segments: buffer energy from flow integration, starting at actual buffer energy
# for idx, h_start in enumerate(sorted(df["hour_start"].unique())):
#     mask = df["hour_start"] == h_start
#     if not mask.any():
#         continue
#     block = df.loc[mask]
#     b_at_start = block["buffer_energy_kwh"].iloc[0]
#     block_heat = block["buffer_heat_kwh_flow"].values.copy()
#     block_heat[0] = 0
#     trajectory = b_at_start + np.cumsum(block_heat)
#     label = "B_a (flow)" if idx == 0 else None
#     ax_buf.plot(block["timestamps"], trajectory, color="tab:orange", linewidth=1.5, label=label)
#     ax_buf.plot(
#         [block["timestamps"].iloc[0], block["timestamps"].iloc[-1]],
#         [trajectory[0], trajectory[-1]],
#         "o", color="tab:orange", markersize=4,
#     )
# # store-flow on secondary y-axis
# ax_buf_y2 = ax_buf.twinx()
# store_flow_col_plot = find_col("store-flow")
# ax_buf_y2.plot(ts, df[store_flow_col_plot].astype(float) / 100, color="gray", linewidth=0.5, alpha=0.5, label="store-flow (GPM)")
# ax_buf_y2.plot(ts, df[dist_flow_col].astype(float) / 100, color="purple", linewidth=0.5, alpha=0.5, label="dist-flow (GPM)")
# ax_buf_y2.plot(ts, df["buffer_lift_C"], color="red", linewidth=0.5, alpha=0.5, label="buffer ΔT (°C)")
# ax_buf_y2.plot(ts, df["buffer_flow_gpm100"] / 100, color="green", linewidth=0.5, alpha=0.5, label="buffer flow (GPM)")
# ax_buf_y2.set_ylabel("GPM / °C")
# ax_buf_y2.tick_params(axis="y", labelcolor="gray")
# # Combined legend
# lines1, labels1 = ax_buf.get_legend_handles_labels()
# lines2, labels2 = ax_buf_y2.get_legend_handles_labels()
# ax_buf.legend(lines1 + lines2, labels1 + labels2, loc="best")
# ax_buf.set_ylabel("Energy (kWh)")
# ax_buf.set_title("Buffer energy")
# ax_buf.grid(True, axis="x", alpha=0.3)

try:
    # Storage from flow integration: single continuous line starting at S_a(0)
    ax.plot(ts, df["storage_from_flow"], color="tab:green", linewidth=1.5, label="S_a flow")

    # Storage actual (continuous)
    # ax.plot(ts, df["storage_energy_kwh"], color="tab:blue", linewidth=1.5, label="S_a")

    # # Expected storage trajectory (buffer from avg temp): commented out
    # if flo_times and flo_load_first_hour:
    #     for idx, h_start in enumerate(segment_start_times):
    #         mask = df["hour_start"] == h_start
    #         if not mask.any():
    #             continue
    #         block = df.loc[mask]
    #         label = "HP_a - B_a(avg_temp) - H_e" if idx == 0 else None
    #         ax.plot(block["timestamps"], block["expected_storage_trajectory"], color="tab:orange", linewidth=1.5, label=label)
    #         ax.plot(
    #             block["timestamps"].iloc[[0, -1]],
    #             block["expected_storage_trajectory"].iloc[[0, -1]],
    #             "o", color="tab:orange", markersize=4,
    #         )

    # Expected storage trajectory with buffer from flow: per-hour segments
    if flo_times and flo_load_first_hour:
        for idx, h_start in enumerate(segment_start_times):
            mask = df["hour_start"] == h_start
            if not mask.any():
                continue
            block = df.loc[mask]
            label = "HP_a - B_a - H_e" if idx == 0 else None
            label_he2 = "HP_a - B_a - H_e2" if idx == 0 else None
            ax.plot(block["timestamps"], block["expected_storage_trajectory_buf_flow"], color="tab:red", linewidth=1.5, label=label)
            ax.plot(
                block["timestamps"].iloc[[0, -1]],
                block["expected_storage_trajectory_buf_flow"].iloc[[0, -1]],
                "o", color="tab:red", markersize=4,
            )
            ax.plot(block["timestamps"], block["expected_storage_trajectory_he2"], color="tab:orange", linewidth=1.5, label=label_he2)
            ax.plot(
                block["timestamps"].iloc[[0, -1]],
                block["expected_storage_trajectory_he2"].iloc[[0, -1]],
                "o", color="tab:orange", markersize=4,
            )

    ax.legend(loc="best")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("HP_a - B_a - H_e compared to S_a")
    ax.grid(True, axis="x", alpha=0.3)
except Exception as e:
    print(f"First plot skipped due to error: {e}")
    ax.set_visible(False)

# Detail plot: HP_a, B_a, H_e per hour (each starting at 0 at hour start)
# Only show hours where we are always in HpOffStoreOff (no background coloring)
first_detail = True
eoh_times = []   # end-of-hour timestamps
eoh_he = []      # end-of-hour H_e values
eoh_he2 = []     # end-of-hour H_e2 values
eoh_da = []      # end-of-hour D_a values
if flo_times and flo_load_first_hour:
    for idx, h_start in enumerate(segment_start_times):
        mask = df["hour_start"] == h_start
        if not mask.any():
            continue
        block = df.loc[mask]
        # HP change within hour (from 0)
        hp_in_hour = block["cumulative_hp_kwh"] - block["cumulative_hp_kwh"].iloc[0]
        # Buffer change within hour (from 0, flow-based)
        buf_in_hour = block["cumulative_buffer_flow_kwh"] - block["cumulative_buffer_flow_kwh"].iloc[0]
        # House expected (prorated, from 0)
        he_in_hour = block["expected_so_far_in_hour"].fillna(0)

        # Distribution change within hour (from 0)
        dist_in_hour = block["cumulative_dist_kwh"] - block["cumulative_dist_kwh"].iloc[0]

        # H_e2 prorated within the hour
        he2_val = he2_per_hour.get(h_start, np.nan)
        elapsed_s_block = (block["timestamps"] - h_start).dt.total_seconds()
        he2_in_hour = he2_val * (elapsed_s_block / 3600.0) if not np.isnan(he2_val) else elapsed_s_block * 0

        # Collect end-of-hour values
        eoh_times.append(block["timestamps"].iloc[-1])
        eoh_he.append(he_in_hour.iloc[-1])
        eoh_he2.append(he2_in_hour.iloc[-1] if not isinstance(he2_in_hour, float) else he2_in_hour)
        eoh_da.append(dist_in_hour.iloc[-1])

        # label_hp = "HP_a" if first_detail else None
        # label_buf = "B_a" if first_detail else None
        # label_he = "H_e" if first_detail else None
        # label_dist = "D_a" if first_detail else None
        # label_he2 = "H_e2" if first_detail else None
        first_detail = False
        # ax_detail.plot(block["timestamps"], hp_in_hour, color="tab:blue", linewidth=1, label=label_hp)
        # ax_detail.plot(block["timestamps"], buf_in_hour, color="tab:orange", linewidth=1, label=label_buf)
        # ax_detail.plot(block["timestamps"], he_in_hour, color="tab:green", linewidth=1, label=label_he)
        # ax_detail.plot(block["timestamps"], dist_in_hour, color="tab:purple", linewidth=1, label=label_dist)
        # ax_detail.plot(block["timestamps"], he2_in_hour, color="tab:red", linewidth=1, label=label_he2)

    # Draw lines connecting end-of-hour values
    if eoh_times:
        ax_detail.plot(eoh_times, eoh_he, color="tab:green", linewidth=1, alpha=0.7, label="House expected (current)")
        ax_detail.plot(eoh_times, eoh_he2, color="tab:red", linewidth=1, alpha=0.7, label="House expected (with solar gains)")
        ax_detail.plot(eoh_times, eoh_da, color="tab:purple", linewidth=1, alpha=0.7, label="Actual distribution heat in")

ax_detail.axhline(0, color="gray", linestyle="--", alpha=0.7)
ax_detail.legend(loc="best")
ax_detail.set_ylabel("Energy (kWh)")
ax_detail.set_title("HP_a, B_a, H_e per hour (starting at 0)")
ax_detail.grid(True, axis="x", alpha=0.3)

ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=ny_tz))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=ny_tz))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
