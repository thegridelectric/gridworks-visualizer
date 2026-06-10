# Realtime Gateway

One central service that replaces the per-house `webinter` WebSocket
processes (`gridworks-scada/gw_spaceheat/webinter`). It consumes all SCADA
telemetry from the GridWorks RabbitMQ broker over **AMQP** and fans it out to
dashboard WebSocket clients, keyed by house short alias.

```
SCADAs --MQTT plugin--> RabbitMQ (amq.topic, vhost hw1__1)
                            |
                            | AMQP, binding key gw.#
                            v
                     realtime gateway ----wss /ws{alias}----> dashboards
```

**Read-only by design.** The gateway never publishes to the broker: no relay
control, no admin dispatch, no snapshot requests. SCADAs push
`snapshot.spaceheat` every ~30s and `layout.lite` on link activation; the
gateway caches the latest per house and pushes them to clients on connect and
on arrival. No SCADA or LTN changes are required.

## WebSocket contract (unchanged from webinter)

- Endpoint: `/ws{short_alias}`, e.g. `/wsoak`
- Server sends `{"type": "status", ...}` (fields: `target_gnode`,
  `thermostat_names`, `layout_loaded`, `snapshot_loaded`, ...) followed by
  `{"type": "mqtt_message", "message_type": "snapshot.spaceheat", "payload": {...}}`
- Client messages `get_status` and `request_snapshot` are answered from the
  cache; everything else is ignored.

## Configuration

Read from the same `.env` as the visualizer API:

| Variable | Default | Meaning |
|----------|---------|---------|
| `VIS_RABBIT_URL` | `amqp://smqPublic:smqPublic@localhost:5672/hw1__1` | AMQP URL incl. vhost |
| `VIS_GATEWAY_PORT` | `8100` | HTTP/WebSocket listen port |

## Running

```bash
cd ~/gridworks-visualizer
python -m gateway          # or ./start_gateway.sh (tmux session "gateway")
```

Health/ops endpoint: `GET http://localhost:8100/gateway/health` lists every
house seen on the broker with snapshot freshness and client counts.

## Tests

`python -m gateway.smoke_test` runs the real server and a real WebSocket
client against injected broker messages (no RabbitMQ needed) and verifies the
full client-facing contract.

## Local development

1. Run a local RabbitMQ: `docker run -d --name rabbit -p 5672:5672 rabbitmq:3`
   (with default vhost, set `VIS_RABBIT_URL=amqp://guest:guest@localhost:5672/`)
2. Start the gateway: `python -m gateway`
3. Publish fake SCADA traffic: `python -m gateway.dev_simulator --houses oak,fir`
4. Connect a client to `ws://localhost:8100/wsoak`

## Deployment / migration (visualizer EC2)

nginx: replace the per-house `location /ws{alias}` blocks with a single
regex block pointing at the gateway:

```nginx
location ~ ^/ws(?<alias>[a-z0-9]+)$ {
    proxy_pass http://127.0.0.1:8100;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 86400;
    proxy_send_timeout 86400;
    proxy_buffering off;
}
```

To migrate house by house: regex locations win over plain prefix locations
in nginx, so convert the block of any house that should *stay* on its
webinter to an exact match (`location = /wsoak { ... }`), which always beats
the regex. Once all houses are cut over, stop the `{house}-webinter` tmux
sessions and delete their clones; new houses then require **zero** gateway
config — they appear as soon as their SCADA publishes to the broker.
