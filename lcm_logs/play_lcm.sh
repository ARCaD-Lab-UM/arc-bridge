#!/usr/bin/env bash
set -euo pipefail

echo "Launching lcm-logplayer-gui with args: $*"
echo "Press Ctrl-C to stop."
exec lcm-logplayer-gui "$@"
