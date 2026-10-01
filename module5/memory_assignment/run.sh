#!/usr/bin/env bash
# Usage: ./run.sh <totalThreads> <blockSize> [textFile]  -> one run
#        ./run.sh                                       -> full sweep:
#            base config, 2 more thread counts, 2 more block sizes
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -gt 0 ]; then
    ./memory_assignment "$@"
    exit
fi

for cfg in "65536 256" "4096 256" "1048576 256" \
           "65536 64" "65536 1024"; do
    echo "===== ./memory_assignment $cfg ====="
    # shellcheck disable=SC2086
    ./memory_assignment $cfg
    echo
done
