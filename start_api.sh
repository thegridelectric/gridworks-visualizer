#!/bin/bash

# Script to start the REST API in a tmux session
SESSION_NAME="api"
RUN_CMD='cd ~/gridworks-web-backend && unset VIRTUAL_ENV && uv run python -m api'
ATTACH=1

if [[ "${1:-}" == "--detach" ]]; then
    ATTACH=0
fi

# If tmux is not installed, run without tmux
if ! command -v tmux &>/dev/null; then
    echo "tmux is not installed. Running without tmux."
    eval "$RUN_CMD"
    exit 0
fi

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    if [[ "$ATTACH" -eq 1 ]]; then
        tmux attach-session -t "$SESSION_NAME"
    fi
else
    echo "Creating new tmux session '$SESSION_NAME'..."

    tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)"
    sleep 0.5

    tmux send-keys -t "$SESSION_NAME" "$RUN_CMD" C-m

    if [[ "$ATTACH" -eq 1 ]]; then
        tmux attach-session -t "$SESSION_NAME"
    else
        echo "Started '$SESSION_NAME' (detached)."
    fi
fi
