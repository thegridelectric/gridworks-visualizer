import bisect
import pickle
import re
import pendulum
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

ZONE_GW_RE = re.compile(r'^zone(\d+)-(.+)-gw-temp$')
ZONE_TEMP_RE = re.compile(r'^zone(\d+)-(.+)-temp$')
HOUSE_ALIAS = "spruce"
# Require max(dist-flow) GPM in [t_start, t_end] to exceed this for a heat-call span to count.
DIST_FLOW_PEAK_MIN_GPM = 0.1
EWMA_ALPHA_GW_TEMP = 0.2
# During a heat call: if gw-temp exceeds predicted setpoint by this many °F (latched), hide setpoint until call ends.
SETPOINT_OVERTEMP_SUPPRESS_DEGF = 2.0
# No active heat: if gw falls this far below predicted setpoint, hide prediction until next gated heat call ends.
SETPOINT_UNDERTEMP_SUPPRESS_DEGF = 2.0
# Heat-call timeline band (blue): vertical extent as a fraction of panel height (axes coordinates, bottom-aligned).
HEAT_CALL_BAND_HEIGHT_FRAC = 0.1
# Unix ms timestamps are interpreted / labeled in this TZ for plotting.
DISPLAY_TZ = 'America/New_York'


def _format_x_unix_ms_ddmm_hh00(ms: float, _pos=None) -> str:
    return pendulum.from_timestamp(ms / 1000.0, tz=DISPLAY_TZ).strftime('%d/%m %H:00')


def ewma_gw_temp_series(
    times: list[float], values: list[float], alpha: float
) -> tuple[list[float], list[float]]:
    """Sort by time and apply EWMA: y[t] = alpha*x[t] + (1-alpha)*y[t-1]; y[0] = x[0]."""
    pairs = sorted(zip(times, values), key=lambda p: p[0])
    st = [p[0] for p in pairs]
    sv = [p[1] for p in pairs]
    if not sv:
        return st, sv
    smoothed = [sv[0]]
    for x in sv[1:]:
        smoothed.append(alpha * x + (1.0 - alpha) * smoothed[-1])
    return st, smoothed


with open('messages.pkl', 'rb') as f:
    messages = pickle.load(f)

data_by_channel: dict[str, dict[str, list[float]]] = {}

for message in messages:
    for r in message.payload['LatestReadingList']:
        if r['ChannelName'] not in data_by_channel:
            print(f'Adding channel: {r["ChannelName"]}')
            data_by_channel[r['ChannelName']] = {
                'times': [],
                'values': [],
            }
        data_by_channel[r['ChannelName']]['times'].append(r['ScadaReadTimeUnixMs'])
        data_by_channel[r['ChannelName']]['values'].append(r['Value'])

print(f'Converted {len(messages)} messages to data_by_channel')

# gw-temp channels are stored as degCx100 (or degCx1000 for BEECH);
# plain -temp channels (non-gw) are stored as degFx1000.
for _name, d in data_by_channel.items():
    if _name.endswith('-gw-temp'):
        if HOUSE_ALIAS == "spruce":
            d['values'] = [(v/100)*9/5+32 for v in d['values']]
        elif HOUSE_ALIAS == "beech":
            d['values'] = [(v/1000)*9/5+32 for v in d['values']]
        else:
            raise ValueError(f'Unknown house alias: {HOUSE_ALIAS}')
    elif _name.endswith('-temp'):
        d['values'] = [v/1000 for v in d['values']]

print(f'Converted gw-temp channels to degF')

for _name, d in data_by_channel.items():
    if _name.endswith('-gw-temp'):
        d['times'], d['values'] = ewma_gw_temp_series(d['times'], d['values'], EWMA_ALPHA_GW_TEMP)

print(f'Applied EWMA (alpha={EWMA_ALPHA_GW_TEMP}) to -gw-temp channels (time-sorted)')

# dist-flow channel is stored as GPMx100
flow_pred_helpers = None
if 'dist-flow' in data_by_channel:
    d = data_by_channel['dist-flow']
    d['values'] = [v / 100 for v in d['values']]

print(f'Converted the dist-flow channel to GPM')


def heat_on_intervals(times: list[float], values: list[float]) -> list[tuple[float, float]]:
    """Contiguous time ranges where heat-call is on (value == 1). Times in ms."""
    if not times:
        return []
    pairs = sorted(zip(times, values), key=lambda p: p[0])
    st = [p[0] for p in pairs]
    sv = [p[1] for p in pairs]
    dt = (st[-1] - st[-2]) if len(st) > 1 else 60_000.0
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for t, v in zip(st, sv):
        on = v >= 0.5
        if on and start is None:
            start = t
        elif not on and start is not None:
            intervals.append((start, t))
            start = None
    if start is not None:
        intervals.append((start, st[-1] + dt))
    return intervals


def heat_call_completed_end_times(times: list[float], values: list[float]) -> list[float]:
    """Timestamps where heat-call transitions off (on→0). Trailing ON with no OFF has no end event."""
    if not times:
        return []
    pairs = sorted(zip(times, values), key=lambda p: p[0])
    st = [p[0] for p in pairs]
    sv = [p[1] for p in pairs]
    ends: list[float] = []
    start: float | None = None
    for t, v in zip(st, sv):
        on = v >= 0.5
        if on and start is None:
            start = t
        elif not on and start is not None:
            ends.append(t)
            start = None
    return ends


def max_dist_flow_in_interval(
    flow_times_sorted: list[float], flow_values: list[float], t_lo: float, t_hi: float
) -> float | None:
    """Max dist-flow sample in [t_lo, t_hi]. flow_times_sorted must be sorted. None if no samples."""
    i0 = bisect.bisect_left(flow_times_sorted, t_lo)
    i1 = bisect.bisect_right(flow_times_sorted, t_hi)
    if i0 >= i1:
        return None
    return max(flow_values[i0:i1])


def heat_calls_gated_by_dist_flow_peak(
    heat_times: list[float],
    heat_values: list[float],
    flow_times_sorted: list[float],
    flow_values: list[float],
    min_peak_gpm: float = DIST_FLOW_PEAK_MIN_GPM,
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Heat-call highlight spans and completion times (OFF) where
    max(dist-flow) over the span is greater than min_peak_gpm.
    Trailing ON (no OFF) can produce a span but never a completion time.
    """
    if not heat_times:
        return [], []
    pairs = sorted(zip(heat_times, heat_values), key=lambda p: p[0])
    st = [p[0] for p in pairs]
    sv = [p[1] for p in pairs]
    dt = (st[-1] - st[-2]) if len(st) > 1 else 60_000.0
    intervals: list[tuple[float, float]] = []
    completions: list[float] = []
    start: float | None = None
    for t, v in zip(st, sv):
        on = v >= 0.5
        if on and start is None:
            start = t
        elif not on and start is not None:
            t0, t1 = start, t
            m = max_dist_flow_in_interval(flow_times_sorted, flow_values, t0, t1)
            if m is not None and m > min_peak_gpm:
                intervals.append((t0, t1))
                completions.append(t1)
            start = None
    if start is not None:
        t0, t1 = start, st[-1] + dt
        m = max_dist_flow_in_interval(flow_times_sorted, flow_values, t0, t1)
        if m is not None and m > min_peak_gpm:
            intervals.append((t0, t1))
    return intervals, completions


def gw_value_at_or_before(gw_times: list[float], gw_values: list[float], t_q: float) -> float | None:
    pairs = sorted(zip(gw_times, gw_values), key=lambda p: p[0])
    st = [p[0] for p in pairs]
    sv = [p[1] for p in pairs]
    i = bisect.bisect_right(st, t_q) - 1
    if i < 0:
        return None
    return sv[i]


def zone_setpoint_series(
    gw_times: list[float],
    gw_values: list[float],
    validated_intervals: list[tuple[float, float]],
    validated_off_times: list[float],
) -> tuple[list[float], list[float]]:
    """NaN until a gated heat-call span has started; then gw-temp at last gated OFF ≤ t."""
    g_pairs = sorted(zip(gw_times, gw_values), key=lambda p: p[0])
    gt = [p[0] for p in g_pairs]

    vp: list[float] = []
    for t_end in validated_off_times:
        gv_at = gw_value_at_or_before(gw_times, gw_values, t_end)
        vp.append(float(gv_at) if gv_at is not None else float('nan'))

    sp_out: list[float] = []
    ej = 0
    last_sp = float('nan')
    for t in gt:
        while ej < len(validated_off_times) and validated_off_times[ej] <= t:
            last_sp = vp[ej]
            ej += 1
        gated_observed = any(ts <= t for ts, _te in validated_intervals)
        sp_out.append(last_sp if gated_observed else float('nan'))
    return gt, sp_out


def _active_heat_interval_index(
    heat_spans: list[tuple[float, float]], t: float
) -> int | None:
    """Index of gated span containing t if t_start <= t < t_end (OFF time excluded)."""
    for k, (a, b) in enumerate(heat_spans):
        if a <= t < b:
            return k
    return None


def apply_setpoint_overtemp_suppression_during_heat(
    gt: list[float],
    gw_values: list[float],
    heat_spans: list[tuple[float, float]],
    sp_base: list[float],
    overtemp_f: float,
) -> tuple[list[float], list[tuple[float, float]]]:
    """
    While inside a gated heat span, if gw-temp >= sp_base + overtemp_f (latched for that span),
    emit NaN until the span ends; then the base series (updated at OFF) applies again.

    Returns (setpoint_series, suppression_spans_for_plot) where each span is [t_when_suppression_began, heat_OFF_time].
    """
    n = len(gt)
    if n != len(gw_values) or n != len(sp_base):
        raise ValueError('gt, gw_values, and sp_base must have equal length')
    out: list[float] = []
    latched = False
    prev_k: int | None = None
    suppressing: list[bool] = []
    for i in range(n):
        t = gt[i]
        k = _active_heat_interval_index(heat_spans, t)
        if k != prev_k:
            latched = False
        prev_k = k

        spb = sp_base[i]
        gv = gw_values[i]

        if k is None:
            suppressing.append(False)
            out.append(spb)
            continue

        if spb != spb:  # nan
            suppressing.append(False)
            out.append(float('nan'))
            continue

        if not latched and gv >= spb + overtemp_f:
            latched = True

        is_suppressing = latched
        suppressing.append(is_suppressing)
        out.append(float('nan') if is_suppressing else spb)

    supp_spans: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not suppressing[i]:
            i += 1
            continue
        k_idx = _active_heat_interval_index(heat_spans, gt[i])
        if k_idx is None:
            i += 1
            continue
        t_lo = gt[i]
        t_hi = heat_spans[k_idx][1]
        j = i + 1
        while j < n and suppressing[j]:
            j += 1
        if t_lo < t_hi:
            supp_spans.append((t_lo, t_hi))
        i = j

    return out, supp_spans


def apply_setpoint_offline_undertemp_suppression(
    gt: list[float],
    gw_values: list[float],
    heat_spans: list[tuple[float, float]],
    validated_off_times: list[float],
    sp_after_overtemp: list[float],
    undertemp_f: float,
) -> tuple[list[float], list[tuple[float, float]]]:
    """
    When not in active heat (outside [heat_start, OFF)), if gw_temp < predicted - undertemp_f
    using predicted ``sp_after_overtemp``, latch NaN output until the next gated heat OFF time > trigger.

    Span list pairs are (trigger_time, clearing_OFF_time); if no OFF exists later, clearing time is ``gt[-1]``.
    """
    n = len(gt)
    if len(gw_values) != n or len(sp_after_overtemp) != n:
        raise ValueError('gt, gw_values, and sp_after_overtemp must align')
    offs = sorted(validated_off_times)

    latch = False
    clear_deadline: float | None = None
    out: list[float] = []
    red_spans: list[tuple[float, float]] = []

    for i in range(n):
        t = gt[i]
        in_heat = _active_heat_interval_index(heat_spans, t) is not None
        pm = sp_after_overtemp[i]
        gv = gw_values[i]

        if latch and clear_deadline is not None and t >= clear_deadline:
            latch = False
            clear_deadline = None

        if not latch and not in_heat and pm == pm and gv < pm - undertemp_f:
            latch = True
            clear_deadline = next((o for o in offs if o > t), None)
            red_hi = clear_deadline if clear_deadline is not None else gt[-1]
            if t <= red_hi:
                red_spans.append((t, red_hi))

        if latch:
            out.append(float('nan'))
        else:
            out.append(pm)

    return out, red_spans


def zone_heat_start_anchor_series(
    gw_times: list[float],
    gw_values: list[float],
    validated_intervals: list[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Piecewise constant: NaN before first gated heat-call start; then gw-temp at last start time ≤ t."""
    g_pairs = sorted(zip(gw_times, gw_values), key=lambda p: p[0])
    gt = [p[0] for p in g_pairs]

    starts = sorted(ts for ts, _te in validated_intervals)
    vp: list[float] = []
    for t_s in starts:
        gv_at = gw_value_at_or_before(gw_times, gw_values, t_s)
        vp.append(float(gv_at) if gv_at is not None else float('nan'))

    out: list[float] = []
    j = 0
    last = float('nan')
    for t in gt:
        while j < len(starts) and starts[j] <= t:
            last = vp[j]
            j += 1
        out.append(last)
    return gt, out


def _timeline_in_any_span(spans: list[tuple[float, float]], t: float) -> bool:
    return any(lo <= t <= hi for lo, hi in spans)


def next_gated_heat_start_after(heat_spans: list[tuple[float, float]], t_exclusive: float) -> float | None:
    sts = sorted(ts for ts, _ in heat_spans if ts > t_exclusive)
    return sts[0] if sts else None


def raw_heat_on_at_or_before(h_times_sorted: list[float], h_vals: list[float], t: float) -> bool:
    """Last heat-call sample at time <= t; treated as OFF if unknown."""
    if not h_times_sorted:
        return False
    j = bisect.bisect_right(h_times_sorted, t) - 1
    if j < 0:
        return False
    return h_vals[j] >= 0.5


def pre_first_gated_ambiguity_spans(
    gw_times_sorted: list[float],
    heat_times: list[float],
    heat_values: list[float],
    first_gated_heat_start: float | None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Before first gated heat-call start: raw heat-call OFF -> green spans; raw ON -> red spans.
    """
    hp = sorted(zip(heat_times, heat_values), key=lambda p: p[0])
    if not gw_times_sorted or not hp:
        return [], []

    ht = [p[0] for p in hp]
    hv = [p[1] for p in hp]

    syn_red: list[tuple[float, float]] = []
    syn_green: list[tuple[float, float]] = []

    idx = 0
    ngw = len(gw_times_sorted)

    while idx < ngw:
        if first_gated_heat_start is not None and gw_times_sorted[idx] >= first_gated_heat_start:
            break
        on = raw_heat_on_at_or_before(ht, hv, gw_times_sorted[idx])
        ist = idx
        idx += 1

        while idx < ngw:
            if first_gated_heat_start is not None and gw_times_sorted[idx] >= first_gated_heat_start:
                break
            if raw_heat_on_at_or_before(ht, hv, gw_times_sorted[idx]) != on:
                break
            idx += 1

        iend = idx - 1

        lo = gw_times_sorted[ist]

        hi = gw_times_sorted[iend]

        if first_gated_heat_start is not None:
            if idx >= ngw or gw_times_sorted[idx] >= first_gated_heat_start:
                hi = max(hi, first_gated_heat_start)

        elif idx < ngw:

            hi = gw_times_sorted[idx]

        if lo <= hi:

            (syn_red if on else syn_green).append((lo, hi))

    return syn_red, syn_green


def mask_zone_heat_start_line_after_highlights(
    gt: list[float],
    h0_raw: list[float],
    overtemp_spans: list[tuple[float, float]],
    undertempo_spans: list[tuple[float, float]],
    heat_spans: list[tuple[float, float]],
    pre_first_syn_spans: list[tuple[float, float]],
) -> list[float]:
    """
    Hide heat-start prediction in real model highlights and pre-first synthetic green/red bands.
    Post-highlight latch only when exiting *real* over/undertemp spans (requires gated heat spans).
    """
    n = len(gt)
    if len(h0_raw) != n:
        raise ValueError('gt and h0_raw must align')

    def in_pre_first_h(t_: float) -> bool:
        return _timeline_in_any_span(pre_first_syn_spans, t_)

    post_block = False
    pending_clear_ts: float | None = None
    in_prev_real = False
    out: list[float] = []

    for i, t in enumerate(gt):
        in_real = _timeline_in_any_span(overtemp_spans, t) or _timeline_in_any_span(undertempo_spans, t)

        if heat_spans and pending_clear_ts is not None and t >= pending_clear_ts:
            post_block = False
            pending_clear_ts = None

        if heat_spans and in_prev_real and not in_real:
            post_block = True
            pending_clear_ts = next_gated_heat_start_after(heat_spans, t)

        suppress = in_pre_first_h(t) or in_real or post_block
        out.append(float('nan') if suppress else h0_raw[i])

        in_prev_real = in_real

    return out


zones_temp_channels = {}
zones_other_temp_channels = {}
for name in data_by_channel:
    if name.endswith('-gw-temp'):
        m = ZONE_GW_RE.match(name)
        if m:
            zones_temp_channels[int(m.group(1))] = name
        else:
            print(f'{name} is not a zone gw channel')
    elif name.endswith('-temp'):
        m = ZONE_TEMP_RE.match(name)
        if m:
            zones_other_temp_channels[int(m.group(1))] = name

print(f'zones_temp_channels: {zones_temp_channels}')
print(f'zones_other_temp_channels: {zones_other_temp_channels}')

flow_t_sorted: list[float] = []
flow_v_sorted: list[float] = []
if 'dist-flow' in data_by_channel:
    _fp = sorted(
        zip(data_by_channel['dist-flow']['times'], data_by_channel['dist-flow']['values']),
        key=lambda p: p[0],
    )
    flow_t_sorted = [p[0] for p in _fp]
    flow_v_sorted = [p[1] for p in _fp]

# Plot the zones (squeeze=False so axes is always a 2D array; then 1D indices work)
nrows = max(1, len(zones_temp_channels))
fig, axes = plt.subplots(nrows, 1, sharex=True, figsize=(10, 8), squeeze=False)
axes = axes.ravel()

for idx, zone in enumerate(sorted(zones_temp_channels)):
    ch_name = zones_temp_channels[zone]
    ax = axes[idx]
    z_times = data_by_channel[ch_name]['times']
    z_values = data_by_channel[ch_name]['values']
    heat_ch = ch_name.replace('-gw-temp', '-heat-call')
    gz = sorted(zip(z_times, z_values), key=lambda p: p[0])
    gt_plot = [p[0] for p in gz]
    gv_plot = [p[1] for p in gz]
    ax.plot(gt_plot, gv_plot, label='gw-temp', zorder=3, alpha=0.5, color='gray')

    if heat_ch in data_by_channel:
        hc = data_by_channel[heat_ch]
        if flow_t_sorted:
            heat_spans, heat_offs = heat_calls_gated_by_dist_flow_peak(
                hc['times'], hc['values'], flow_t_sorted, flow_v_sorted
            )
        else:
            heat_spans = heat_on_intervals(hc['times'], hc['values'])
            heat_offs = heat_call_completed_end_times(hc['times'], hc['values'])
        heat_band_trans = blended_transform_factory(ax.transData, ax.transAxes)
        for t0, t1 in heat_spans:
            ax.axvspan(
                t0,
                t1,
                ymin=0,
                ymax=HEAT_CALL_BAND_HEIGHT_FRAC,
                transform=heat_band_trans,
                alpha=0.45,
                color='tab:blue',
                zorder=0,
                linewidth=0,
            )
        _gt_ord, sp_base = zone_setpoint_series(z_times, z_values, heat_spans, heat_offs)
        sp, overtemp_spans = apply_setpoint_overtemp_suppression_during_heat(
            _gt_ord, gv_plot, heat_spans, sp_base, SETPOINT_OVERTEMP_SUPPRESS_DEGF
        )
        sp, undertemp_spans = apply_setpoint_offline_undertemp_suppression(
            _gt_ord,
            gv_plot,
            heat_spans,
            heat_offs,
            sp,
            SETPOINT_UNDERTEMP_SUPPRESS_DEGF,
        )
        first_hs = min(ts for ts, _ in heat_spans) if heat_spans else None
        syn_red, syn_green = pre_first_gated_ambiguity_spans(
            _gt_ord, hc['times'], hc['values'], first_hs
        )
        for s0, s1 in syn_green:
            ax.axvspan(s0, s1, alpha=0.22, color='tab:green', zorder=0, linewidth=0)
        for s0, s1 in syn_red:
            ax.axvspan(s0, s1, alpha=0.2, color='tab:red', zorder=0, linewidth=0)
        for s0, s1 in overtemp_spans:
            ax.axvspan(s0, s1, alpha=0.2, color='tab:red', zorder=1, linewidth=0)
        for r0, r1 in undertemp_spans:
            ax.axvspan(r0, r1, alpha=0.22, color='tab:green', zorder=2, linewidth=0)
        ax.plot(_gt_ord, sp, label='Heat stops (pred)', color='tab:orange', linestyle='--', zorder=3, linewidth=2.5)
        _gt_h0, h0_raw = zone_heat_start_anchor_series(z_times, z_values, heat_spans)
        pre_first_syn = syn_red + syn_green
        h0 = mask_zone_heat_start_line_after_highlights(
            _gt_h0,
            h0_raw,
            overtemp_spans,
            undertemp_spans,
            heat_spans,
            pre_first_syn,
        )
        ax.plot(
            _gt_h0,
            h0,
            label='Heat starts (pred)',
            color='tab:cyan',
            linestyle='--',
            zorder=3,
            linewidth=2.5,
        )
    if zone in zones_other_temp_channels and HOUSE_ALIAS != "spruce":
        other_name = zones_other_temp_channels[zone]
        ax.plot(data_by_channel[other_name]['times'],
                data_by_channel[other_name]['values'],
                label='temp', color='tab:purple', zorder=3)
    leg_handles, _leg_labels = ax.get_legend_handles_labels()
    if heat_ch in data_by_channel:
        leg_handles.extend(
            [
                Patch(
                    facecolor='tab:red',
                    alpha=0.2,
                    edgecolor='none',
                    label='Setpoint increased, below setpoint',
                ),
                Patch(
                    facecolor='tab:green',
                    alpha=0.22,
                    edgecolor='none',
                    label='Setpoint decreased, above setpoint',
                ),
            ]
        )
    if leg_handles:
        ax.legend(handles=leg_handles, loc='upper right', fontsize=8)
    ax.set_ylabel(f'Zone {zone} Temp (°F)')
    ax.set_title(f'Zone {zone}')

# Plot dist-flow (disabled for now)
# ax = axes[-1]
# flow = data_by_channel['dist-flow']
# ax.plot(flow['times'], flow['values'], color='tab:blue')
# ax.set_ylabel('Dist Flow (GPM)')
# ax.set_title('Dist Flow')

if zones_temp_channels:
    axes[0].set_xlim(
        left=min(data_by_channel[zones_temp_channels[z]]['times'][0] for z in zones_temp_channels),
        right=max(data_by_channel[zones_temp_channels[z]]['times'][-1] for z in zones_temp_channels),
    )
else:
    ft = data_by_channel['dist-flow']['times']
    axes[0].set_xlim(left=ft[0], right=ft[-1])

for ax in axes.ravel():
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_format_x_unix_ms_ddmm_hh00))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.show()