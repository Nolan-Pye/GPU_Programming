# CUDA Advanced Libraries Assignment

**Nolan Pye — EN605.617, Module 9**  
**Repository:** [GPU Programming, main branch](https://github.com/Nolan-Pye/GPU_Programming/tree/main/module9/advanced_libraries_assignment)

## Goal and use case

I built a Monte Carlo savings simulator to estimate the chance of reaching a
financial goal with different stock/bond mixes. The default run models
1,000,000 possible 30-year futures for someone saving $500 per month toward
$1,000,000. It compares six portfolios using stock and Treasury returns from
1928–2025 and reports success rates and balance statistics.

## Library integration

The program combines **cuRAND from Module 7** and **Thrust from Module 8**.
A seeded Philox generator creates the returns with `curandGenerateNormal`; an
alternate model uses `curandGenerateUniform` to choose historical years.
cuRAND writes directly into two `thrust::device_vector<float>` objects.

Thrust consumes the returns in place: `transform` computes balances,
`count_if` measures success, reductions calculate statistics, and `sort` finds
percentiles. All portfolios reuse the same futures for a fair comparison.
Year-major storage also gives neighboring GPU threads neighboring values.

## Results and validation

In the default run, 100% stocks reached the goal in **58.3%** of futures versus
**31.9%** for 60/40. At 2,000,000 futures, Thrust took 42.9 ms versus 2,264.8
ms on the CPU, a **53x speedup**. Checks compare cuRAND's sample means with the
source data and GPU results with both a CPU version and an exact formula. All
checks pass in [`results.txt`](results.txt).

## Challenges and triumphs

My first challenge was connecting cuRAND's raw-pointer interface to Thrust's
containers without an extra copy. The historical model also had to keep stock
and bond returns from the same year paired; one random year now drives both
lookups.

Large runs initially exhausted memory, so I added clear exception handling.
CUDA startup distorted short timings, which an untimed warm-up fixed. My main
triumph is a useful, validated application that keeps both libraries on the
GPU. The model remains intentionally simple: it ignores taxes, fees, inflation,
and serial correlation and applies savings annually.

## Build and run

```bash
./build.sh
./run.sh
./run.sh model=history
./run.sh all
```

The repository-root `assignment_config.yaml` builds the program and runs all
graded scenarios automatically.
