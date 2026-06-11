#!/bin/bash

./start_gateway.sh --detach
./start_api.sh --detach

echo
echo "Both services running in tmux."
echo "  tmux attach -t gateway"
echo "  tmux attach -t api"
