#!/usr/bin/env bash
# Usage: ./run.sh <totalThreads> <blockSize>
# Each invocation appends its timing results to results.csv.
set -euo pipefail
cd "$(dirname "$0")"

./assignment.exe "$@"
