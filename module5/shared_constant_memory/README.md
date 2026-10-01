# Shared and Constant Memory Examples

## Constant memory

| File | Demonstrates |
|------|--------------|
| `constant_memory.cu` | The same bitwise loop using literal values, `__constant__ static const` scalars, and `__device__` (global) scalars, with timing for each |
| `constant_memory2.cu` | A `__constant__` array vs. a `__device__` array, both filled from the host with `cudaMemcpyToSymbol` |

Sizes can be overridden at compile time:

```bash
nvcc -DKERNEL_LOOP=1024 -DNUM_ELEMENTS=16384 -DWORK_SIZE=64 -o constant_memory constant_memory.cu
```

- `KERNEL_LOOP`: iterations of the kernel loop. In `constant_memory2.cu` it is also
  the `__constant__` array length, so it must be at most 16384 (the 64KB constant memory limit).
- `NUM_ELEMENTS`: threads launched for the timing comparison.
- `WORK_SIZE`: elements checked against the host-computed result.

`../run_practicals.sh` builds and runs both files with their default sizes, then with smaller and larger sizes.
