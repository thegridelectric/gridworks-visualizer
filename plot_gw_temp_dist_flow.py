import bisect
import math
import pickle
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

NY_TZ = ZoneInfo('America/New_York')

FLOW_ZERO_EPS = 0.1
ZONE_GW_RE = re.compile(r'^zone(\d+)-(.+)-gw-temp$')


def _flow_zero_crossing_anchor_pairs(flow_times, flow_values, eps=FLOW_ZERO_EPS):
    """For each flow-off transition, return (t_zero, t_last_nonzero).

    t_last_nonzero is the orange-highlight dist-flow point (last |flow|>eps before off).
    Predicted set switches at t_last_nonzero to gw-temp at that time until the next such point.
    """
    order = sorted(range(len(flow_times)), key=lambda i: flow_times[i])
    ft = [flow_times[i] for i in order]
    fv = [flow_values[i] for i in order]
    pairs = []
    for i in range(len(fv)):
        if abs(fv[i]) > eps:
            continue
        if i > 0 and abs(fv[i - 1]) <= eps:
            continue
        j = i - 1
        while j >= 0 and abs(fv[j]) <= eps:
            j -= 1
        if j < 0:
            continue
        pairs.append((ft[i], ft[j]))
    return pairs


def _pre_zero_flow_original_indices(flow_times, flow_values, eps=FLOW_ZERO_EPS):
    """Original-array indices of the last |flow|>eps sample before each flow-off transition."""
    order = sorted(range(len(flow_times)), key=lambda i: flow_times[i])
    ft = [flow_times[i] for i in order]
    fv = [flow_values[i] for i in order]
    out = set()
    for i in range(len(fv)):
        if abs(fv[i]) > eps:
            continue
        if i > 0 and abs(fv[i - 1]) <= eps:
            continue
        j = i - 1
        while j >= 0 and abs(fv[j]) <= eps:
            j -= 1
        if j < 0:
            continue
        out.add(order[j])
    return out


def _gw_sorted_with_setpoint_anchor_indices(gw_times, gw_values, zero_anchor_pairs):
    """Sorted (gt, gv) and indices of gw samples used to lock piecewise setpoint (same rule as predicted curve)."""
    gw_pairs = sorted(zip(gw_times, gw_values), key=lambda p: p[0])
    gt = [p[0] for p in gw_pairs]
    gv = [p[1] for p in gw_pairs]
    if not zero_anchor_pairs:
        return gt, gv, set()
    anchor_times = [p[1] for p in zero_anchor_pairs]
    anchor_idx = set()
    for t_anchor in anchor_times:
        k = bisect.bisect_right(gt, t_anchor) - 1
        if k >= 0:
            anchor_idx.add(k)
    return gt, gv, anchor_idx


def _predicted_set_values(gw_times, gw_values, zero_anchor_pairs):
    """Piecewise constant: at each orange (last-nonzero) flow time, lock gw-temp until the next."""
    if not zero_anchor_pairs:
        return [float('nan')] * len(gw_times)
    gw_pairs = sorted(zip(gw_times, gw_values), key=lambda p: p[0])
    gt = [p[0] for p in gw_pairs]
    gv = [p[1] for p in gw_pairs]
    anchor_times = [p[1] for p in zero_anchor_pairs]
    locked = []
    for t_anchor in anchor_times:
        k = bisect.bisect_right(gt, t_anchor) - 1
        locked.append(float('nan') if k < 0 else gv[k])
    out = []
    for t in gw_times:
        k = bisect.bisect_right(anchor_times, t) - 1
        if k < 0:
            out.append(float('nan'))
        else:
            out.append(locked[k])
    return out


with open('messages.pkl', 'rb') as f:
    messages = pickle.load(f)

data_by_channel = {}

for message in messages:
    for r in message.payload['LatestReadingList']:
        if r['ChannelName'] not in data_by_channel:
            data_by_channel[r['ChannelName']] = {
                'times': [],
                'values': [],
            }
        data_by_channel[r['ChannelName']]['times'].append(r['ScadaReadTimeUnixMs'])
        data_by_channel[r['ChannelName']]['values'].append(r['Value'])

for data in data_by_channel.values():
    data['times'] = [
        datetime.fromtimestamp(ms / 1000.0, tz=NY_TZ) for ms in data['times']
    ]

for _name, d in data_by_channel.items():
    if _name.endswith('-gw-temp'):
        d['values'] = [(v / 100.0) * (9.0 / 5.0) + 32.0 for v in d['values']]

flow_pred_helpers = None
if 'dist-flow' in data_by_channel:
    d = data_by_channel['dist-flow']
    # Stored as GPM×100 (see visualizer_api channel scaling hints).
    d['values'] = [v / 100.0 for v in d['values']]
    flow_pred_helpers = _flow_zero_crossing_anchor_pairs(d['times'], d['values'])

flat_times = []
for _name in sorted(k for k in data_by_channel if k.endswith('-gw-temp')):
    flat_times.extend(data_by_channel[_name]['times'])
if 'dist-flow' in data_by_channel:
    flat_times.extend(data_by_channel['dist-flow']['times'])

plt.rcParams['timezone'] = 'America/New_York'

zone_gw_by_x = {}
other_gw_channels = []
for name in data_by_channel:
    if not name.endswith('-gw-temp'):
        continue
    m = ZONE_GW_RE.match(name)
    if m:
        zone_gw_by_x[int(m.group(1))] = name
    else:
        other_gw_channels.append(name)
other_gw_channels.sort()

gw_rows = []
for x in sorted(zone_gw_by_x.keys()):
    gw_rows.append(('zone', x, zone_gw_by_x[x]))
if other_gw_channels:
    gw_rows.append(('other', None, tuple(other_gw_channels)))
if not gw_rows:
    gw_rows.append(('empty', None, None))

nrows = len(gw_rows) + 1
fig_h = min(3.0 * nrows + 1.0, 36.0)
fig, axes = plt.subplots(
    nrows,
    1,
    sharex=True,
    figsize=(11, fig_h),
    layout='constrained',
    squeeze=False,
)
axes = axes.ravel()

for row, (kind, x, payload) in enumerate(gw_rows):
    ax = axes[row]
    ax.set_ylabel('Temperature [degF]')
    if kind == 'empty':
        ax.text(0.5, 0.5, 'No -gw-temp channels', ha='center', va='center', transform=ax.transAxes)
        continue
    if kind == 'zone':
        channel_name = payload
        data = data_by_channel[channel_name]
        if flow_pred_helpers is not None:
            gt, gv, anchor_idx = _gw_sorted_with_setpoint_anchor_indices(
                data['times'], data['values'], flow_pred_helpers
            )
            ax.plot(data['times'], data['values'], label=channel_name)
            if anchor_idx:
                ai = sorted(anchor_idx)
                ax.scatter(
                    [gt[i] for i in ai],
                    [gv[i] for i in ai],
                    c='red',
                    s=48,
                    alpha=0.95,
                    edgecolors='black',
                    linewidths=0.6,
                    zorder=5,
                )
            pred = _predicted_set_values(data['times'], data['values'], flow_pred_helpers)
            ax.plot(
                data['times'],
                pred,
                color='darkred',
                linestyle='--',
                alpha=0.9,
                label=f'zone{x}-predicted-set',
            )
        else:
            ax.plot(data['times'], data['values'], label=channel_name)
        ax.set_title(f'Zone {x}')
        ax.legend()
    else:
        for channel_name in payload:
            data = data_by_channel[channel_name]
            ax.plot(data['times'], data['values'], label=channel_name)
        ax.set_title('Other gw-temp')
        ax.legend()

ax_flow = axes[-1]
if 'dist-flow' in data_by_channel:
    flow_data = data_by_channel['dist-flow']
    ft = flow_data['times']
    fv = flow_data['values']
    pre_idx = _pre_zero_flow_original_indices(ft, fv)
    ax_flow.plot(ft, fv, color='tab:blue', label='dist-flow')
    if pre_idx:
        pi = sorted(pre_idx)
        ax_flow.scatter(
            [ft[i] for i in pi],
            [fv[i] for i in pi],
            c='tab:orange',
            s=48,
            alpha=0.95,
            edgecolors='black',
            linewidths=0.6,
            zorder=5,
            label='last non-zero before off',
        )
    ax_flow.legend()
else:
    ax_flow.text(0.5, 0.5, 'dist-flow not in data', ha='center', va='center', transform=ax_flow.transAxes)
ax_flow.set_ylabel('dist-flow [GPM]')
ax_flow.set_xlabel('Time')

if flat_times:
    hour_span = max(
        (max(flat_times) - min(flat_times)).total_seconds() / 3600.0,
        1e-9,
    )
    hour_step = max(1, math.ceil(hour_span / 8))
else:
    hour_step = 1
x_formatter = mdates.DateFormatter('%d/%m %H:00', tz=NY_TZ)

for ax in axes:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_step, tz=NY_TZ))
    ax.xaxis.set_major_formatter(x_formatter)
    ax.grid(True, axis='both', linestyle='--', alpha=0.35)

fig.autofmt_xdate(rotation=0, ha='center')

plt.show()
