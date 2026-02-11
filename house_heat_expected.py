"""
Standalone plot comparing hourly expected house heat (H_e, H_e2) to actual
distribution heat (D_a).

Only requires two CSV files — no database or FLO solver needed:
  1. Hourly electricity CSV   ({HOUSE_ALIAS}_electricity_use_*.csv)
  2. Weather station CSV       (weather_KMEMILLI18.csv_processed.csv)
"""

HOUSE_ALIAS = "beech"
WEATHER_CSV = "weather_KMEMILLI18.csv_processed.csv"
ALPHA = 8.6
BETA = -0.14
GAMMA = 0.005

import numpy as np
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "data"

# ---- Load hourly electricity CSV ----
elec_matches = sorted(data_dir.glob(f"{HOUSE_ALIAS}_electricity_use_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not elec_matches:
    raise FileNotFoundError(f'No hourly electricity CSV matching "{HOUSE_ALIAS}_electricity_use_*.csv"')
elec_path = elec_matches[0]
print(f"Using hourly: {elec_path.name}")
edf = pd.read_csv(elec_path)
edf["hour_start"] = pd.to_datetime(edf["hour_start"])
if edf["hour_start"].dt.tz is None:
    edf["hour_start"] = edf["hour_start"].dt.tz_localize("America/New_York", ambiguous="infer")
else:
    edf["hour_start"] = edf["hour_start"].dt.tz_convert("America/New_York")
edf = edf.sort_values("hour_start").reset_index(drop=True)

# Compute H_e from ALPHA, BETA, GAMMA constants and oat_f, ws_mph from CSV
edf["computed_expected_house_kwh"] = (
    ALPHA + BETA * edf["oat_f"]
    + GAMMA * edf["ws_mph"] * (65 - edf["oat_f"])
)
print(f"H_e: {len(edf)} hours (ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA})")

# ---- Load weather CSV (solar irradiance) ----
def _load_weather_solar():
    weather_path = data_dir / WEATHER_CSV
    if not weather_path.exists():
        print(f"Weather CSV not found: {weather_path}")
        return {}, {}, {}
    wdf = pd.read_csv(weather_path)
    wdf["hour_start"] = pd.to_datetime(wdf["Datetime"])
    if wdf["hour_start"].dt.tz is None:
        wdf["hour_start"] = wdf["hour_start"].dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        wdf["hour_start"] = wdf["hour_start"].dt.tz_convert("America/New_York")
    solar_map = dict(zip(wdf["hour_start"], wdf["Solar_W/m2"]))
    oat_map = dict(zip(wdf["hour_start"], wdf["Temperature_F"]))
    ws_map = dict(zip(wdf["hour_start"], wdf["Wind_Speed_Mph"]))
    print(f"Weather solar: loaded {len(solar_map)} hours from {WEATHER_CSV}")
    return solar_map, oat_map, ws_map

_solar_map, _weather_oat, _weather_ws = _load_weather_solar()

segment_start_times = list(edf["hour_start"])

# ---- Red curve: BETA, GAMMA from top of file; ALPHA shifted to best fit dist_kwh ----
_valid = edf.dropna(subset=["dist_kwh"])
_weather_term = BETA * _valid["oat_f"] + GAMMA * _valid["ws_mph"] * (65 - _valid["oat_f"])
_alpha_fit = (_valid["dist_kwh"] - _weather_term).mean()
print(f"Red curve: ALPHA adjusted from {ALPHA:.4f} to {_alpha_fit:.4f} (BETA={BETA}, GAMMA={GAMMA})")
he_csv_fit_per_hour = dict(zip(
    edf["hour_start"],
    _alpha_fit + BETA * edf["oat_f"] + GAMMA * edf["ws_mph"] * (65 - edf["oat_f"])
))

# ---- Solar data ----
edf["solar"] = edf["hour_start"].map(_solar_map).fillna(0)

# ---- Compute end-of-hour D_a from dist_kwh in hourly CSV ----
da_end_all = {}
for _, row in edf.iterrows():
    val = row.get("dist_kwh", np.nan)
    if not pd.isna(val):
        da_end_all[row["hour_start"]] = val

# Build a lookup from hour_start to edf row data
edf_lookup = {}
for _, row in edf.iterrows():
    edf_lookup[row["hour_start"]] = row

# ---- Rolling day-by-day fit for orange and blue curves ----
# For each day, fit using up to 5 previous days of data (not including current day).
# Orange: fit alpha/beta/gamma, no solar
# Blue: same + fit delta for solar
TRAILING_DAYS = 5

# Get unique days in the data
all_hours_with_da = sorted(da_end_all.keys())
all_days = sorted(set(h.floor("D") for h in all_hours_with_da))
print(f"Rolling fit: {len(all_days)} days, {len(all_hours_with_da)} hours with D_a")

he_opt_per_hour = {}  # orange curve
he2_per_hour = {}     # blue curve

for day_idx, current_day in enumerate(all_days):
    # Trailing window: previous days only (not including current day)
    window_start = current_day - pd.Timedelta(days=TRAILING_DAYS)
    trailing_hours = [h for h in all_hours_with_da
                      if h.floor("D") >= window_start and h.floor("D") < current_day]

    trailing_days_count = len(set(h.floor("D") for h in trailing_hours))
    if trailing_days_count < TRAILING_DAYS:
        print(f"  Day {current_day.strftime('%Y-%m-%d')}: skipped (only {trailing_days_count} trailing days, need >= {TRAILING_DAYS})")
        continue

    # Build training data from trailing hours
    train_rows = []
    for h in trailing_hours:
        if h not in edf_lookup or h not in da_end_all:
            continue
        row = edf_lookup[h]
        train_rows.append({
            "oat_f": row["oat_f"], "ws_mph": row["ws_mph"],
            "solar": row["solar"], "da": da_end_all[h],
            "beta_csv": row["beta"], "gamma_csv": row["gamma"],
        })

    if len(train_rows) < TRAILING_DAYS * 20:
        print(f"  Day {current_day.strftime('%Y-%m-%d')}: skipped (only {len(train_rows)} matched trailing hours, need >= {TRAILING_DAYS * 20})")
        continue

    train_df = pd.DataFrame(train_rows)

    # --- Orange curve: fit alpha/beta/gamma without solar ---
    X = np.column_stack([
        np.ones(len(train_df)),
        train_df["oat_f"].values,
        train_df["ws_mph"].values * (65 - train_df["oat_f"].values),
    ])
    y = train_df["da"].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a_opt, b_opt, g_opt = coeffs

    # --- Blue curve: fit delta for solar on top of orange ---
    he_opt_train = a_opt + b_opt * train_df["oat_f"] + g_opt * train_df["ws_mph"] * (65 - train_df["oat_f"])
    residual = train_df["da"] - he_opt_train
    solar_vals = train_df["solar"].values
    denom = (solar_vals ** 2).sum()
    delta = (residual.values * solar_vals).sum() / denom if denom > 1e-12 else 0.0

    # Apply to current day's hours
    current_day_hours = [h for h in all_hours_with_da if h.floor("D") == current_day]
    for h in current_day_hours:
        if h not in edf_lookup:
            continue
        row = edf_lookup[h]
        he_opt_per_hour[h] = a_opt + b_opt * row["oat_f"] + g_opt * row["ws_mph"] * (65 - row["oat_f"])
        he2_per_hour[h] = he_opt_per_hour[h] + delta * row["solar"]

    print(f"  Day {current_day.strftime('%Y-%m-%d')}: fitted from {len(train_rows)} trailing hours "
          f"(a={a_opt:.3f}, d={delta:.5f})")

# ---- Global best fit (all data) ----
global_fit_per_hour = {}
_global_rows = []
for h in all_hours_with_da:
    if h not in edf_lookup:
        continue
    row = edf_lookup[h]
    _global_rows.append({
        "oat_f": row["oat_f"], "ws_mph": row["ws_mph"], "da": da_end_all[h],
    })
if len(_global_rows) >= 3:
    _gdf = pd.DataFrame(_global_rows).dropna()
    if len(_gdf) >= 3:
        _X_global = np.column_stack([
            np.ones(len(_gdf)),
            _gdf["oat_f"].values,
            _gdf["ws_mph"].values * (65 - _gdf["oat_f"].values),
        ])
        _y_global = _gdf["da"].values
        try:
            _g_coeffs, _, _, _ = np.linalg.lstsq(_X_global, _y_global, rcond=None)
            a_g, b_g, g_g = _g_coeffs
            print(f"Global fit: a={a_g:.4f}, b={b_g:.4f}, g={g_g:.5f} (from {len(_gdf)} hours)")
            for h in all_hours_with_da:
                if h not in edf_lookup:
                    continue
                row = edf_lookup[h]
                global_fit_per_hour[h] = a_g + b_g * row["oat_f"] + g_g * row["ws_mph"] * (65 - row["oat_f"])
        except np.linalg.LinAlgError as e:
            print(f"Global fit failed: {e}")

# ---- Plot ----
fig, (ax, ax_err) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
ny_tz = pytz.timezone("America/New_York")

eoh_times = []
eoh_he = []
eoh_he_opt = []
eoh_he2 = []
eoh_da = []
eoh_global = []

for h_start in segment_start_times:
    da_val = da_end_all.get(h_start, np.nan)
    if pd.isna(da_val):
        continue

    eoh_times.append(h_start + pd.Timedelta(hours=1))
    eoh_he.append(he_csv_fit_per_hour.get(h_start, np.nan))
    eoh_he_opt.append(he_opt_per_hour.get(h_start, np.nan))
    eoh_he2.append(he2_per_hour.get(h_start, np.nan))
    eoh_da.append(da_val)
    eoh_global.append(global_fit_per_hour.get(h_start, np.nan))

eoh_he = np.array(eoh_he, dtype=float)
eoh_he_opt = np.array(eoh_he_opt, dtype=float)
eoh_da = np.array(eoh_da, dtype=float)
eoh_global = np.array(eoh_global, dtype=float)

# Draw lines connecting end-of-hour values
if len(eoh_times):
    ax.plot(eoh_times, eoh_he, color="tab:red", linewidth=2, alpha=0.7, label="Predicted (current house parameters)")
    ax.plot(eoh_times, eoh_he_opt, color="tab:blue", linewidth=2, alpha=0.7, label=f"Predicted (fit over trailing {TRAILING_DAYS} days)")
    # ax.plot(eoh_times, eoh_he2, color="tab:blue", linewidth=2, alpha=1, label="Predicted (fit, with solar gains)")
    ax.plot(eoh_times, eoh_global, color="tab:green", linewidth=2, alpha=0.7, label="Predicted (best fit, all data)")
    ax.plot(eoh_times, eoh_da, color="gray", linewidth=1, alpha=0.4, label="Actual distribution heat in")

ax.legend(loc="best")
ax.set_ylabel("Energy (kWh)")
# ax.set_ylim(0, 10)
ax.set_title(HOUSE_ALIAS)

# ---- Error plot (predicted - actual) ----
if len(eoh_times):
    err_he = eoh_he - eoh_da
    err_he_opt = eoh_he_opt - eoh_da
    err_global = eoh_global - eoh_da
    # Compute RMSE (ignoring NaNs)
    rmse_he = np.sqrt(np.nanmean(err_he ** 2))
    rmse_he_opt = np.sqrt(np.nanmean(err_he_opt ** 2))
    rmse_global = np.sqrt(np.nanmean(err_global ** 2))
    ax_err.plot(eoh_times, err_he, color="tab:red", linewidth=1, alpha=0.7,
                label=f"RMSE={rmse_he:.2f} kWh")
    ax_err.plot(eoh_times, err_he_opt, color="tab:blue", linewidth=1, alpha=0.7,
                label=f"RMSE={rmse_he_opt:.2f} kWh")
    ax_err.plot(eoh_times, err_global, color="tab:green", linewidth=1, alpha=0.7,
                label=f"RMSE={rmse_global:.2f} kWh")
    ax_err.axhline(0, color="gray", linestyle="--", alpha=0.5)

ax_err.legend(loc="best")
ax_err.set_ylabel("Error (kWh)")
ax_err.set_xlabel("Time")

ax_err.xaxis.set_major_locator(mdates.AutoDateLocator(tz=ny_tz))
ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=ny_tz))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
