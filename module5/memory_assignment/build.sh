#!/usr/bin/env bash
# Builds ./memory_assignment from a clean state.
set -euo pipefail
cd "$(dirname "$0")"

make clean
make
