"""
Standalone plot comparing hourly expected house heat (H_e, H_e2) to actual
distribution heat (D_a).

Requires two CSV files — no database or FLO solver needed:
  1. Hourly electricity CSV   ({HOUSE_ALIAS}_electricity_use_*.csv)
  2. Weather station CSV      (weather_KMEMILLI18.csv_processed.csv) — used to fill NaN in oat_f, ws_mph
"""

PARAMETERS = {
    'oak': {
        'ALPHA': 7.6,
        'BETA': -0.13,
        'GAMMA': 0.0024,
    },
    'beech': {
        'ALPHA': 8.6,
        'BETA': -0.14,
        'GAMMA': 0.005,
    },
}

HOUSE_ALIAS = "oak"
WEATHER_CSV = "weather_KMEMILLI18.csv_processed.csv"
ALPHA = PARAMETERS[HOUSE_ALIAS]['ALPHA']
BETA = PARAMETERS[HOUSE_ALIAS]['BETA']
GAMMA = PARAMETERS[HOUSE_ALIAS]['GAMMA']
FIND_WINDOW_LENGTH = False  # If True, sweep 1–20 days and plot RMSE; if False, use 10 days

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

# Fill NaN in oat_f, ws_mph using weather CSV; fallback to ffill/bfill
def _load_weather_for_fill():
    weather_path = data_dir / WEATHER_CSV
    if not weather_path.exists():
        print(f"Weather CSV not found: {weather_path}")
        return {}, {}
    wdf = pd.read_csv(weather_path)
    wdf["hour_start"] = pd.to_datetime(wdf["Datetime"])
    if wdf["hour_start"].dt.tz is None:
        wdf["hour_start"] = wdf["hour_start"].dt.tz_localize("America/New_York", ambiguous=False)
    else:
        wdf["hour_start"] = wdf["hour_start"].dt.tz_convert("America/New_York")
    oat_map = dict(zip(wdf["hour_start"], wdf["Temperature_F"]))
    ws_map = dict(zip(wdf["hour_start"], wdf["Wind_Speed_Mph"]))
    print(f"Weather: loaded {len(oat_map)} hours from {WEATHER_CSV} (for filling oat_f, ws_mph)")
    return oat_map, ws_map

_weather_oat, _weather_ws = _load_weather_for_fill()
edf["oat_f"] = edf["oat_f"].fillna(edf["hour_start"].map(_weather_oat)).ffill().bfill().fillna(65.0)
edf["ws_mph"] = edf["ws_mph"].fillna(edf["hour_start"].map(_weather_ws)).ffill().bfill().fillna(0.0)

# Compute H_e from ALPHA, BETA, GAMMA constants and oat_f, ws_mph from CSV
edf["computed_expected_house_kwh"] = (
    ALPHA + BETA * edf["oat_f"]
    + GAMMA * edf["ws_mph"] * (65 - edf["oat_f"])
)
print(f"H_e: {len(edf)} hours (ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA})")

# # ---- Load weather CSV (solar irradiance) ----
# def _load_weather_solar():
#     weather_path = data_dir / WEATHER_CSV
#     if not weather_path.exists():
#         print(f"Weather CSV not found: {weather_path}")
#         return {}, {}, {}
#     wdf = pd.read_csv(weather_path)
#     wdf["hour_start"] = pd.to_datetime(wdf["Datetime"])
#     if wdf["hour_start"].dt.tz is None:
#         wdf["hour_start"] = wdf["hour_start"].dt.tz_localize("America/New_York", ambiguous=False)
#     else:
#         wdf["hour_start"] = wdf["hour_start"].dt.tz_convert("America/New_York")
#     solar_map = dict(zip(wdf["hour_start"], wdf["Solar_W/m2"]))
#     oat_map = dict(zip(wdf["hour_start"], wdf["Temperature_F"]))
#     ws_map = dict(zip(wdf["hour_start"], wdf["Wind_Speed_Mph"]))
#     print(f"Weather solar: loaded {len(solar_map)} hours from {WEATHER_CSV}")
#     return solar_map, oat_map, ws_map
# _solar_map, _weather_oat, _weather_ws = _load_weather_solar()

# ---- Load 60-second telemetry CSV (for setpoints) ----
# Pattern: oak_60s_2026-02-02-01_00-2026-02-10-05_00.csv (exclude rolling_params files)
telemetry_matches = sorted([p for p in data_dir.glob(f"{HOUSE_ALIAS}_*s_*.csv") if "params" not in p.name], 
                          key=lambda p: p.stat().st_mtime, reverse=True)
if telemetry_matches:
    telem_path = telemetry_matches[0]
    print(f"Using telemetry: {telem_path.name}")
    df_telem = pd.read_csv(telem_path, skiprows=[0])
    df_telem["timestamps"] = pd.to_datetime(df_telem["timestamps"])
    if df_telem["timestamps"].dt.tz is None:
        df_telem["timestamps"] = df_telem["timestamps"].dt.tz_localize("America/New_York", ambiguous="infer")
    else:
        df_telem["timestamps"] = df_telem["timestamps"].dt.tz_convert("America/New_York")
    setpoint_cols = [c for c in df_telem.columns if c.endswith("-set")]
    print(f"Setpoint columns: {setpoint_cols}")
else:
    df_telem = None
    setpoint_cols = []
    print("No telemetry CSV found for setpoints")

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

# # ---- Solar data ----
# edf["solar"] = edf["hour_start"].map(_solar_map).fillna(0)

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

all_hours_with_da = sorted(da_end_all.keys())
all_days = sorted(set(h.floor("D") for h in all_hours_with_da))
print(f"Rolling fit: {len(all_days)} days, {len(all_hours_with_da)} hours with D_a")

def _rolling_fit_for_window(tw, debug_skip_reasons=False):
    """Run rolling day-by-day fit for a given trailing window length.
    Returns (fits, fitted_days, params_list) where params_list is [(day, alpha, beta, gamma), ...].
    """
    fits = {}
    params_list = []
    fitted_days = 0
    skip_reasons = {"trailing_days": 0, "train_rows": 0, "dropna": 0, "finite": 0, "lstsq": 0, "coeffs": 0}
    for current_day in all_days:
        window_start = current_day - pd.Timedelta(days=tw)
        trailing_hours = [h for h in all_hours_with_da
                          if h.floor("D") >= window_start and h.floor("D") < current_day]

        train_rows = []
        for h in trailing_hours:
            if h not in edf_lookup or h not in da_end_all:
                continue
            row = edf_lookup[h]
            train_rows.append({
                "oat_f": row["oat_f"], "ws_mph": row["ws_mph"],
                "da": da_end_all[h],
            })

        if not train_rows:
            skip_reasons["train_rows"] += 1
            continue
        train_df = pd.DataFrame(train_rows).dropna(subset=["oat_f", "ws_mph", "da"])
        train_df = train_df[np.isfinite(train_df["oat_f"]) & np.isfinite(train_df["ws_mph"]) & np.isfinite(train_df["da"])]
        # Require minimum 48 hours (2 days) of valid data; allow gaps in the 10-day window
        MIN_TRAIN_HOURS = 48
        if len(train_df) < MIN_TRAIN_HOURS:
            skip_reasons["train_rows"] += 1
            continue

        X = np.column_stack([
            np.ones(len(train_df)),
            train_df["oat_f"].values,
            train_df["ws_mph"].values * (65 - train_df["oat_f"].values),
        ])
        y = train_df["da"].values
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            skip_reasons["finite"] += 1
            continue
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-10)
        except np.linalg.LinAlgError:
            skip_reasons["lstsq"] += 1
            continue
        if not np.all(np.isfinite(coeffs)):
            skip_reasons["coeffs"] += 1
            continue
        a_opt, b_opt, g_opt = coeffs
        params_list.append((current_day, a_opt, b_opt, g_opt))

        current_day_hours = [h for h in all_hours_with_da if h.floor("D") == current_day]
        for h in current_day_hours:
            if h not in edf_lookup:
                continue
            row = edf_lookup[h]
            fits[h] = a_opt + b_opt * row["oat_f"] + g_opt * row["ws_mph"] * (65 - row["oat_f"])
        fitted_days += 1
    if skip_reasons and any(v > 0 for v in skip_reasons.values()):
        print(f"  Days skipped: {dict(skip_reasons)}")
    return fits, fitted_days, params_list

# def _rolling_fit_with_solar(tw):
#     """Rolling fit with solar gains: fit alpha/beta/gamma, then delta for solar."""
#     ...
#     return fits, fitted_days

if FIND_WINDOW_LENGTH:
    # ---- Sweep trailing window lengths 1–20 days, compute RMSE for each ----
    RMSE_SWEEP = range(1, 21)
    rolling_fits = {}
    print(f"Sweeping {len(list(RMSE_SWEEP))} window lengths...")

    rolling_params = {}
    for tw in RMSE_SWEEP:
        print(f"  Fitting window={tw} days...", end=" ", flush=True)
        fits, fitted_days, params_list = _rolling_fit_for_window(tw)
        rolling_fits[tw] = fits
        rolling_params[tw] = params_list
        print(f"{fitted_days} days fitted, {len(fits)} hours predicted")

    # Compute RMSE for each window length
    print("Computing RMSE for each window length...")
    rmse_by_tw = {}
    for tw in RMSE_SWEEP:
        errors = []
        for h in all_hours_with_da:
            pred = rolling_fits[tw].get(h, np.nan)
            if not np.isnan(pred):
                errors.append(pred - da_end_all[h])
        if errors:
            rmse_by_tw[tw] = np.sqrt(np.mean(np.array(errors) ** 2))
            print(f"  Window {tw:2d} days: RMSE={rmse_by_tw[tw]:.3f} kWh ({len(errors)} hours)")

    # Pick smallest window with the best (lowest) RMSE
    best_rmse = min(rmse_by_tw.values())
    BEST_WINDOW = min(tw for tw, r in rmse_by_tw.items() if r == best_rmse)
    print(f"\nBest trailing window: {BEST_WINDOW} days (RMSE={best_rmse:.3f} kWh)")
    best_fit = rolling_fits[BEST_WINDOW]
    params_list = rolling_params[BEST_WINDOW]

    # ---- Figure 1: RMSE vs window length ----
    fig_rmse, ax_rmse = plt.subplots(figsize=(8, 4))
    tws = sorted(rmse_by_tw.keys())
    rmses = [rmse_by_tw[tw] for tw in tws]
    ax_rmse.plot(tws, rmses, "o-", color="tab:blue")
    ax_rmse.axvline(BEST_WINDOW, color="tab:red", linestyle="--", alpha=0.7, label=f"Best: {BEST_WINDOW} days")
    ax_rmse.set_xlabel("Trailing window (days)")
    ax_rmse.set_ylabel("RMSE (kWh)")
    ax_rmse.set_title(f"{HOUSE_ALIAS} — RMSE vs trailing window length")
    ax_rmse.set_xticks(tws)
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3)
    plt.tight_layout()
else:
    BEST_WINDOW = 10
    print(f"Using default trailing window: {BEST_WINDOW} days")
    best_fit, _fd, params_list = _rolling_fit_for_window(BEST_WINDOW)
    print(f"  {_fd} days fitted, {len(best_fit)} hours predicted")

# ---- Export fitted parameters (alpha, beta, gamma) per day ----
params_df = pd.DataFrame(
    [{"day": d.strftime("%Y-%m-%d"), "alpha": round(a, 3), "beta": round(b, 3), "gamma": round(g, 6)}
     for d, a, b, g in params_list]
)
params_csv = data_dir / f"{HOUSE_ALIAS}_rolling_params_{BEST_WINDOW}d.csv"
params_df.to_csv(params_csv, index=False)
print(f"Exported {len(params_list)} days of parameters to {params_csv.name}")

# ---- Figure: OAT and rolling beta over time ----
ny_tz = pytz.timezone("America/New_York")
fig_oat, (ax_oat, ax_set) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)

# --- Top plot: Average OAT over each day's trailing window (same window used to fit beta) ---
avg_oat_by_day = []
for d, a, b, g in params_list:
    window_start = d - pd.Timedelta(days=BEST_WINDOW)
    mask = (edf["hour_start"] >= window_start) & (edf["hour_start"] < d)
    avg_oat_by_day.append(edf.loc[mask, "oat_f"].mean())

beta_days = [d for d, a, b, g in params_list]
ax_oat.plot(beta_days, avg_oat_by_day, color="tab:orange", linewidth=2, alpha=0.8, marker="s", markersize=4,
            label=f"OAT (average past {BEST_WINDOW} days)")

ax_oat.set_ylabel("OAT (°F)")
ax_oat.legend(loc="upper left")

ax_beta = ax_oat.twinx()
beta_vals = [b for d, a, b, g in params_list]
ax_beta.plot(beta_days, beta_vals, color="tab:blue", linewidth=2, alpha=0.8, marker="o", markersize=4,
            label=f"Beta (trained on past {BEST_WINDOW} days)")
ax_beta.set_ylabel(f"β (rolling {BEST_WINDOW}d fit)")
ax_beta.legend(loc="upper right")
ax_oat.set_title(HOUSE_ALIAS)

# --- Bottom plot: Setpoints over time ---
if df_telem is not None and setpoint_cols:
    for col in setpoint_cols:
        ax_set.plot(df_telem["timestamps"], df_telem[col], label=col.replace("-set", ""), alpha=0.8, linewidth=1)
    ax_set.set_ylabel("Setpoint (°F)")
    ax_set.legend(loc="upper left", fontsize=8, ncol=2)
else:
    ax_set.text(0.5, 0.5, "No setpoint data available", ha="center", va="center", transform=ax_set.transAxes)

ax_set.set_xlabel("Time")
ax_set.xaxis.set_major_locator(mdates.AutoDateLocator(tz=ny_tz))
ax_set.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=ny_tz))
fig_oat.autofmt_xdate()
plt.tight_layout()

# # ---- Rolling fit with solar gains (same window) ----
# print(f"Fitting with solar gains (trailing {BEST_WINDOW} days)...", flush=True)
# best_fit_solar, _fd_solar = _rolling_fit_with_solar(BEST_WINDOW)
# print(f"  {_fd_solar} days fitted, {len(best_fit_solar)} hours predicted (with solar)")

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
    _gdf = _gdf[np.isfinite(_gdf["oat_f"]) & np.isfinite(_gdf["ws_mph"]) & np.isfinite(_gdf["da"])]
    if len(_gdf) >= 3:
        _X_global = np.column_stack([
            np.ones(len(_gdf)),
            _gdf["oat_f"].values,
            _gdf["ws_mph"].values * (65 - _gdf["oat_f"].values),
        ])
        _y_global = _gdf["da"].values
        if np.all(np.isfinite(_X_global)) and np.all(np.isfinite(_y_global)):
            try:
                _g_coeffs, _, _, _ = np.linalg.lstsq(_X_global, _y_global, rcond=1e-10)
                a_g, b_g, g_g = _g_coeffs
                if np.all(np.isfinite(_g_coeffs)):
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
eoh_da = []
eoh_global = []
eoh_best = []
# eoh_best_solar = []

for h_start in segment_start_times:
    da_val = da_end_all.get(h_start, np.nan)
    if pd.isna(da_val):
        continue

    eoh_times.append(h_start + pd.Timedelta(hours=1))
    eoh_he.append(he_csv_fit_per_hour.get(h_start, np.nan))
    eoh_da.append(da_val)
    eoh_global.append(global_fit_per_hour.get(h_start, np.nan))
    eoh_best.append(best_fit.get(h_start, np.nan))
    # eoh_best_solar.append(best_fit_solar.get(h_start, np.nan))

eoh_he = np.array(eoh_he, dtype=float)
eoh_da = np.array(eoh_da, dtype=float)
eoh_global = np.array(eoh_global, dtype=float)
eoh_best = np.array(eoh_best, dtype=float)
# eoh_best_solar = np.array(eoh_best_solar, dtype=float)

# Restrict all curves to hours where the blue (rolling fit) curve has predictions
eoh_times_full = list(eoh_times)
valid_mask = ~np.isnan(eoh_best)
n_removed = (~valid_mask).sum()
n_kept = valid_mask.sum()

removed_times = [t for t, v in zip(eoh_times_full, valid_mask) if not v]
kept_times = [t for t, v in zip(eoh_times_full, valid_mask) if v]
first_fitted_day = params_list[0][0] if params_list else None

print(f"Hours with dist_kwh: {len(eoh_times_full)} total")
print(f"Hours with rolling fit prediction: {n_kept}")
print(f"Hours removed (no blue curve prediction): {n_removed}")
if removed_times:
    print(f"  Removed range: {min(removed_times).strftime('%Y-%m-%d %H:%M')} to {max(removed_times).strftime('%Y-%m-%d %H:%M')}")
if kept_times:
    print(f"  Kept range:     {min(kept_times).strftime('%Y-%m-%d %H:%M')} to {max(kept_times).strftime('%Y-%m-%d %H:%M')}")
if first_fitted_day:
    print(f"  Reason: Rolling fit needs {BEST_WINDOW} trailing days. First day fitted: {first_fitted_day.strftime('%Y-%m-%d')}. "
          f"Hours before that have no prediction.")

eoh_times = kept_times
eoh_he = eoh_he[valid_mask]
eoh_da = eoh_da[valid_mask]
eoh_global = eoh_global[valid_mask]
eoh_best = eoh_best[valid_mask]
# eoh_best_solar = eoh_best_solar[valid_mask]
print(f"Plotting {len(eoh_times)} hours (after removing {n_removed} hours without rolling fit)")

# Draw lines connecting end-of-hour values
if len(eoh_times):
    ax.plot(eoh_times, eoh_he, color="tab:red", linewidth=2, alpha=0.7, label="Predicted (current house parameters)")
    ax.plot(eoh_times, eoh_best, color="tab:blue", linewidth=2, alpha=0.7,
            label=f"Predicted (fit, trailing {BEST_WINDOW} days)")
    # ax.plot(eoh_times, eoh_best_solar, color="tab:cyan", linewidth=2, alpha=0.7,
    #         label=f"Predicted (fit + solar, trailing {BEST_WINDOW} days)")
    # ax.plot(eoh_times, eoh_global, color="tab:green", linewidth=2, alpha=0.7, label="Predicted (fit on all data, incl. future)")
    ax.plot(eoh_times, eoh_da, color="gray", linewidth=1, alpha=0.4, label="Actual distribution heat in")

ax.legend(loc="best")
ax.set_ylabel("Energy (kWh)")
# ax.set_ylim(0, 10)
ax.set_title(HOUSE_ALIAS)

# ---- Error plot (predicted - actual) ----
if len(eoh_times):
    err_he = eoh_he - eoh_da
    err_best = eoh_best - eoh_da
    # err_global = eoh_global - eoh_da
    rmse_he = np.sqrt(np.nanmean(err_he ** 2))
    rmse_best = np.sqrt(np.nanmean(err_best ** 2))
    # rmse_global = np.sqrt(np.nanmean(err_global ** 2))
    ax_err.plot(eoh_times, err_he, color="tab:red", linewidth=1, alpha=0.7,
                label=f"RMSE={rmse_he:.2f} kWh")
    ax_err.plot(eoh_times, err_best, color="tab:blue", linewidth=1, alpha=0.7,
                label=f"RMSE={rmse_best:.2f} kWh ({BEST_WINDOW}d)")
    # err_best_solar = eoh_best_solar - eoh_da
    # rmse_best_solar = np.sqrt(np.nanmean(err_best_solar ** 2))
    # ax_err.plot(eoh_times, err_best_solar, color="tab:cyan", linewidth=1, alpha=0.7,
    #             label=f"RMSE={rmse_best_solar:.2f} kWh ({BEST_WINDOW}d + solar)")
    # ax_err.plot(eoh_times, err_global, color="tab:green", linewidth=1, alpha=0.7,
    #             label=f"RMSE={rmse_global:.2f} kWh")
    ax_err.axhline(0, color="gray", linestyle="--", alpha=0.5)

ax_err.legend(loc="best")
ax_err.set_ylabel("Error (kWh)")
ax_err.set_xlabel("Time")

ax_err.xaxis.set_major_locator(mdates.AutoDateLocator(tz=ny_tz))
ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=ny_tz))
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
