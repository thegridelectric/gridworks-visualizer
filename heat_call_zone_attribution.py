import pickle
import re
import pendulum
import matplotlib.pyplot as plt

EPSILON = 0.2

ZONE_GW_RE = re.compile(r'^zone(\d+)-(.+)-gw-temp$')


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

print(f'Converted {len(messages)} messages to data_by_channel')

# gw-temp channels are stored as degCx100
for _name, d in data_by_channel.items():
    if _name.endswith('-gw-temp'):
        d['values'] = [(v/100)*9/5+32 for v in d['values']]

print(f'Converted {len(data_by_channel)} channels to degF')

# dist-flow channel is stored as GPMx100
flow_pred_helpers = None
if 'dist-flow' in data_by_channel:
    d = data_by_channel['dist-flow']
    d['values'] = [v / 100 for v in d['values']]

print(f'Converted the dist-flow channel to GPM')

zones_temp_channels = {}
for name in data_by_channel:
    if not name.endswith('-gw-temp'):
        continue
    m = ZONE_GW_RE.match(name)
    if m:
        zones_temp_channels[int(m.group(1))] = name
    else:
        print(f'{name} is not a zone gw channel')

print(f'zones_temp_channels: {zones_temp_channels}')

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

# Plot the zones and dist-flow
fig, axes = plt.subplots(len(zones_temp_channels)+1, 1, sharex=True, figsize=(10, 8))

# Plot each zone's gw-temp on its own subplot, highlight heat calls,
# and mark min / max-after-min per heat call.
for idx, zone in enumerate(sorted(zones_temp_channels)):
    ch_name = zones_temp_channels[zone]
    ax = axes[idx]
    z_times = data_by_channel[ch_name]['times']
    z_values = data_by_channel[ch_name]['values']
    ax.plot(z_times, z_values)
    ax.set_ylabel(f'Zone {zone} GW Temp (°F)')
    ax.set_title(f'Zone {zone} GW Temp')

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
        max_idx = max(range(len(in_period)), key=lambda i: in_period[i][1])
        max_t, max_v = in_period[max_idx]
        color = 'red' if (max_v - before_v) > EPSILON else 'green'
        ax.axvspan(start_ts, end_ts, color=color, alpha=0.1)
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

plt.tight_layout()
plt.show()