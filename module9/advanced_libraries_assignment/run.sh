#!/usr/bin/env bash
# Usage: ./run.sh [key=value ...]  -> one run (defaults: $500 a month for
#                                     30 years, goal $1,000,000)
#        ./run.sh all              -> every scenario below, as in
#                                     results.txt
# Exits non-zero if any run reports a failed check.
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck source=win_cuda_env.sh
source ./win_cuda_env.sh

if [ "${1:-}" != "all" ]; then
    ./savings_sim "$@"
    exit
fi

status=0
for args in "" \
            "model=history" \
            "monthly=250 years=40" \
            "monthly=1000 years=20" \
            "futures=10000" \
            "futures=100000" \
            "futures=2000000"
do
    echo "===== ./savings_sim $args ====="
    # shellcheck disable=SC2086
    ./savings_sim $args || status=1
    echo
done
exit $status
