#!/usr/bin/env bash
# Builds ./savings_sim from a clean state.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=win_cuda_env.sh
source ./win_cuda_env.sh

make clean
make
