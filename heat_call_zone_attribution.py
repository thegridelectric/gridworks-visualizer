import pickle
import re
import pendulum
import matplotlib.pyplot as plt

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

# Plot the zones and dist-flow
fig, axes = plt.subplots(len(zones_temp_channels)+1, 1, sharex=True, figsize=(10, 8))

# Plot each zone's gw-temp on its own subplot
for idx, zone in enumerate(sorted(zones_temp_channels)):
    ch_name = zones_temp_channels[zone]
    ax = axes[idx]
    ax.plot(data_by_channel[ch_name]['times'], data_by_channel[ch_name]['values'])
    ax.set_ylabel(f'Zone {zone} GW Temp (°F)')
    ax.set_title(f'Zone {zone} GW Temp')

# Plot dist-flow
ax = axes[-1]
flow = data_by_channel['dist-flow']
ax.plot(flow['times'], flow['values'], color='tab:orange')
ax.set_ylabel('Dist Flow (GPM)')
ax.set_title('Dist Flow')

axes[0].set_xlim(left=min(data_by_channel[zones_temp_channels[z]]['times'][0] for z in zones_temp_channels),
                 right=max(data_by_channel[zones_temp_channels[z]]['times'][-1] for z in zones_temp_channels))
axes[-1].set_xlabel('Time')

plt.tight_layout()
plt.show()