#!/usr/bin/env bash
# Module 3 practical activities: build and run hello-world, blocks, and grids,
# each across several thread counts, block sizes, and grid dimensions.
set -e
cd "$(dirname "$0")"

nvcc -o hello-world hello-world.cu
nvcc -o blocks blocks.cu
nvcc -o grids grids.cu

echo "===== hello-world (5 configurations) ====="
./hello-world

echo "===== blocks (5 configurations) ====="
./blocks

echo "===== grids (6 configurations, incl. original 1x4 and 2x2) ====="
./grids
