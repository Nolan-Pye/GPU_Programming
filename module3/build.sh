#!/usr/bin/env bash
# Builds assignment.exe. Also resets results.csv so a fresh grading
# session starts with a clean set of recorded runs.
set -euo pipefail
cd "$(dirname "$0")"

make clean
make
