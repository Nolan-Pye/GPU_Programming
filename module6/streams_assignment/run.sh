#!/usr/bin/env bash
# Usage: ./run.sh <blockX> <blockY> [batchSize]  -> one run
#        ./run.sh                               -> full sweep:
#            base config, 4 more block shapes, 2 more batch sizes
# Every run executes each variant once untimed, then 3 timed times.
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -gt 0 ]; then
    ./streams_assignment "$@"
    exit
fi

for cfg in "16 16 32" \
           "8 8 32" "32 32 32" "32 8 32" "128 1 32" \
           "16 16 8" "16 16 64"; do
    echo "===== ./streams_assignment $cfg ====="
    # shellcheck disable=SC2086
    ./streams_assignment $cfg
    echo
done
