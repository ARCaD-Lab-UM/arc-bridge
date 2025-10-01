#!/usr/bin/env bash
set -euo pipefail

echo "Starting lcm-logger in: $(pwd)"
echo "Press Ctrl-C to stop."
# ctrl-c can directly affect the lcm-logger process
exec lcm-logger
