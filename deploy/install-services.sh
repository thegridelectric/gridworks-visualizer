#!/bin/bash
# Install systemd units for the API and gateway. Run from the repo root:
#   ./deploy/install-services.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
USER="$(whoami)"

if [[ ! -x "$REPO/.venv/bin/python" ]]; then
    echo "Missing $REPO/.venv — run 'uv sync' in the repo first." >&2
    exit 1
fi

if [[ ! -f "$REPO/.env" ]]; then
    echo "Missing $REPO/.env — copy template.env and fill in credentials." >&2
    exit 1
fi

for svc in gridworks-api gridworks-gateway; do
    sed "s|@REPO@|$REPO|g; s|@USER@|$USER|g" \
        "$REPO/deploy/$svc.service" | sudo tee "/etc/systemd/system/$svc.service" >/dev/null
    echo "Installed /etc/systemd/system/$svc.service"
done

sudo systemctl daemon-reload
sudo systemctl enable gridworks-api gridworks-gateway
echo
echo "Enabled gridworks-api and gridworks-gateway (start on boot)."
echo "  sudo systemctl start gridworks-api gridworks-gateway"
echo "  sudo systemctl status gridworks-api gridworks-gateway"
echo "  journalctl -u gridworks-api -f"
echo "  journalctl -u gridworks-gateway -f"
