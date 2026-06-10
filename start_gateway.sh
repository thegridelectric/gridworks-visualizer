#!/bin/bash

# Script to start the realtime gateway in a tmux session
SESSION_NAME="gateway"

# If tmux is not installed, run without tmux
if ! command -v tmux &>/dev/null; then
    echo "tmux is not installed. Running without tmux."
    if command -v gw &>/dev/null; then
        gw
    fi
    cd ~/gridworks-visualizer && python -m gateway
    exit 0
fi

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t "$SESSION_NAME"
else
    echo "Creating new tmux session '$SESSION_NAME'..."

    tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)"
    sleep 0.5

    # Run 'gw' command (your alias)
    tmux send-keys -t "$SESSION_NAME" "gw" C-m
    sleep 0.5

    tmux send-keys -t "$SESSION_NAME" "cd ~/gridworks-visualizer && python -m gateway" C-m

    tmux attach-session -t "$SESSION_NAME"
fi
