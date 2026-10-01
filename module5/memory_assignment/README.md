# Module 5 – CUDA Memory Assignment: Letter Histogram

Counts how often each letter a–z appears (case-insensitive) in
*War and Peace* (`war_and_peace.txt`, public domain, Project Gutenberg;
the Gutenberg license header and footer are trimmed off before counting).

Every kernel runs the **same** per-byte code, `countByte()`: look the byte
up in a 256-entry table that maps it to a letter bin (or "not a letter"),
then atomically add 1 to that bin. Only the memory holding the text, the
table, and the histogram changes between variants, so the timings compare
memory spaces like-for-like. Threads walk the text with a grid-stride loop,
so any thread count covers the whole book.

| Variant | Text | Lookup table | Histogram | Rubric item |
|---------|------|--------------|-----------|-------------|
| `cpu` | host | host | host | Host (reference) |
| `host (mapped)` | pinned, mapped **host** memory | global | global | Host |
| `global` | **global** | global | global | Global |
| `constant` | global | **`__constant__`** | global | Constant |
| `shared` | global | global | per-block **shared**, merged at end | Shared |
| `register` | 16-byte loads into **registers** | global | global | Register |
| `combined` | registers | constant | shared | All in one kernel |

The program also:

- checks every GPU histogram against the CPU one (exact match → `PASS`;
  any mismatch → `FAIL` and a non-zero exit code);
- times host→device copies of the text from pageable vs. pinned memory;
- prints each variant's average time over 20 runs (after a warm-up) and its
  speedup vs. the global-memory variant;
- prints a two-column letter-frequency bar chart.

## What the timings show (GTX 1650 Ti)

- **Shared memory wins by far (≈3–10×).** With a global histogram, every
  letter is an atomic on one of only 26 addresses, so millions of threads
  queue on the same memory. A per-block shared histogram keeps those
  atomics on-chip and leaves only 26 global atomics per block.
- **Constant and register variants ≈ global** because global-atomic
  contention dominates; making the reads faster doesn't help.
- **Combined is slower than shared alone.** Threads in a warp look up
  *different* bytes, and constant memory serializes reads of different
  addresses within a warp; it is fastest when all threads read the same one.
- **Host (mapped)** pays PCIe latency on every read of the text.

## Files

| File | Contents |
|------|----------|
| `main.cu` | CLI parsing, buffers, timing, verification, report |
| `histogram_kernels.cu` / `.cuh` | Constant memory, kernels, launchers |
| `text_utils.cpp` / `.h` | File loading, lookup table, CPU histogram, chart |
| `common.h` | `CUDA_CHECK`, tunable constants, `LaunchConfig` |
| `war_and_peace.txt` | Input text |
| `assignment.md` | Write-up: results, figures, discussion |
| `plot_results.py` | Builds the figures from `results.txt` (`uv run --with matplotlib python plot_results.py`) |

## Build and run

```bash
./build.sh                                   # -> ./memory_assignment
./run.sh <totalThreads> <blockSize> [textFile]
./run.sh                                     # full sweep (5 configurations)
```

Defaults are 65,536 threads, a block size of 256, and `war_and_peace.txt`.
A thread count not divisible by the block size is rounded up to whole
blocks.

The sweep, also listed in the repository-root `assignment_config.yaml`:

| Threads | Block size | Purpose |
|---------|------------|---------|
| 65,536 | 256 | base |
| 4,096 | 256 | thread count #2 (~816 bytes per thread) |
| 1,048,576 | 256 | thread count #3 (~4 bytes per thread) |
| 65,536 | 64 | block size #2 |
| 65,536 | 1024 | block size #3 |
