import pickle
import re
import pendulum
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

EPSILON_DEFAULT = 0.2
EPSILON_MIN = -1.0
EPSILON_MAX = 5.0

GW_EWMA_ALPHA = 0.1

ZONE_GW_RE = re.compile(r'^zone(\d+)-(.+)-gw-temp$')
ZONE_TEMP_RE = re.compile(r'^zone(\d+)-(.+)-temp$')


with open('messages.pkl', 'rb') as f:
    messages = pickle.load(f)

data_by_channel = {}

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
        # d['values'] = [(v/100)*9/5+32 for v in d['values']] # SPRUCE
        d['values'] = [(v/1000)*9/5+32 for v in d['values']] # BEECH
    elif _name.endswith('-temp'):
        d['values'] = [v/1000 for v in d['values']]

print(f'Converted gw-temp channels to degF')

# dist-flow channel is stored as GPMx100
flow_pred_helpers = None
if 'dist-flow' in data_by_channel:
    d = data_by_channel['dist-flow']
    d['values'] = [v / 100 for v in d['values']]

print(f'Converted the dist-flow channel to GPM')

# Smooth -gw-temp channels with an exponential weighted moving average:
# y[i] = alpha * x[i] + (1 - alpha) * y[i-1], with y[0] = x[0].
for _name, d in data_by_channel.items():
    if not _name.endswith('-gw-temp'):
        continue
    if not d['values']:
        continue
    smoothed = [d['values'][0]]
    for v in d['values'][1:]:
        smoothed.append(GW_EWMA_ALPHA * v + (1 - GW_EWMA_ALPHA) * smoothed[-1])
    d['values'] = smoothed

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

# Identify periods where dist-flow >= 0.1 GPM.
# Each period is bounded by the last <0.1 sample before the run and the first <0.1 sample after.
flow_periods = []
flow_times = data_by_channel['dist-flow']['times']
flow_values = data_by_channel['dist-flow']['values']
in_period = False
start_ts = None
last_low_ts = None
for t, v in zip(flow_times, flow_values):
    if v >= 0.1:
        if not in_period:
            start_ts = last_low_ts if last_low_ts is not None else t
            in_period = True
    else:
        if in_period:
            flow_periods.append((start_ts, t))
            in_period = False
        last_low_ts = t
if in_period:
    flow_periods.append((start_ts, flow_times[-1]))

print(f'Identified {len(flow_periods)} dist-flow >= 0.1 GPM periods')

# Per-zone epsilon defaults (eps1, eps2, ...)
EPSILONS = {zone: EPSILON_DEFAULT for zone in zones_temp_channels}

# Plot the zones and dist-flow
fig, axes = plt.subplots(len(zones_temp_channels)+1, 1, sharex=True, figsize=(10, 8))

# Per-zone heat call entries: list of (axvspan_obj, before_v, max_v)
zone_entries = {zone: [] for zone in zones_temp_channels}

for idx, zone in enumerate(sorted(zones_temp_channels)):
    ch_name = zones_temp_channels[zone]
    ax = axes[idx]
    z_times = data_by_channel[ch_name]['times']
    z_values = data_by_channel[ch_name]['values']
    ax.plot(z_times, z_values, label='gw-temp')
    if zone in zones_other_temp_channels:
        other_name = zones_other_temp_channels[zone]
        ax.plot(data_by_channel[other_name]['times'],
                data_by_channel[other_name]['values'],
                label='temp', color='tab:purple')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_ylabel(f'Zone {zone} Temp (°F)')
    ax.set_title(f'Zone {zone}')

    for start_ts, end_ts in flow_periods:
        in_period = [(t, v) for t, v in zip(z_times, z_values) if start_ts <= t <= end_ts]
        if not in_period:
            ax.axvspan(start_ts, end_ts, color='green', alpha=0.1)
            continue
        before = next(((t, v) for t, v in zip(reversed(z_times), reversed(z_values)) if t < start_ts), None)
        if before is None:
            ax.axvspan(start_ts, end_ts, color='green', alpha=0.1)
            continue
        before_t, before_v = before
        after = next(((t, v) for t, v in zip(z_times, z_values) if t > end_ts), None)
        candidates = in_period + ([after] if after is not None else [])
        max_idx = max(range(len(candidates)), key=lambda i: candidates[i][1])
        max_t, max_v = candidates[max_idx]
        span = ax.axvspan(start_ts, end_ts, color='green', alpha=0.1)
        zone_entries[zone].append((span, before_v, max_v))
        ax.scatter([before_t, max_t], [before_v, max_v], color='black', s=20, zorder=5)

# Plot dist-flow
ax = axes[-1]
flow = data_by_channel['dist-flow']
ax.plot(flow['times'], flow['values'], color='tab:orange')
ax.set_ylabel('Dist Flow (GPM)')
ax.set_title('Dist Flow')
for start_ts, end_ts in flow_periods:
    ax.axvspan(start_ts, end_ts, color='green', alpha=0.1)

axes[0].set_xlim(left=min(data_by_channel[zones_temp_channels[z]]['times'][0] for z in zones_temp_channels),
                 right=max(data_by_channel[zones_temp_channels[z]]['times'][-1] for z in zones_temp_channels))
axes[-1].set_xlabel('Time')

def apply_eps(zone):
    eps = EPSILONS[zone]
    for span, before_v, max_v in zone_entries[zone]:
        color = 'red' if max_v > before_v + eps else 'green'
        span.set_facecolor(color)
        span.set_edgecolor(color)

for zone in zones_temp_channels:
    apply_eps(zone)

# Reserve room at the bottom for one slider per zone
n_sliders = len(zones_temp_channels)
slider_h = 0.025
slider_pad = 0.01
bottom_reserved = 0.06 + (slider_h + slider_pad) * n_sliders
fig.subplots_adjust(bottom=bottom_reserved, hspace=0.5)

sliders = {}
for i, zone in enumerate(sorted(zones_temp_channels)):
    ax_s = fig.add_axes([0.15, 0.02 + i * (slider_h + slider_pad), 0.7, slider_h])
    s = Slider(ax_s, f'eps{zone}', EPSILON_MIN, EPSILON_MAX, valinit=EPSILONS[zone])

    def make_cb(z):
        def cb(val):
            EPSILONS[z] = val
            apply_eps(z)
            fig.canvas.draw_idle()
        return cb

    s.on_changed(make_cb(zone))
    sliders[zone] = s

plt.show()