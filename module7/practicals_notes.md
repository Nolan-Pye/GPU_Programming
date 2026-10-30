# Module 7 Practical Activities – cuFFT, cuBLAS, cuSPARSE, cuSOLVER, cuRAND

Run everything with `./run_practicals.sh`. The full log from a GTX 1650 Ti
(Turing, SM 7.5, 4 GB, CUDA 13.0) is in `practicals_output.txt`.

## The common library pattern

All five libraries follow the same lifecycle, and recognizing it is most of the
lesson:

1. **Create a context/handle/plan**: `cufftPlan2d`, `cublasInit`/`cublasCreate`,
   `cusparseCreate`, `cusolverDnCreate`, `curandCreateGenerator`. This is the
   expensive, one-time step (library load, kernel selection, workspace
   allocation). Create it once and reuse it.
2. **Put data on the device** with `cudaMalloc`/`cudaMemcpy` (or library helpers
   like `cublasSetMatrix`). Library calls take **device pointers**.
3. **Ask how much scratch memory is needed** where the API needs it
   (`*_bufferSize`), allocate it, and pass it in.
4. **Call the routine.** It is asynchronous on the handle's stream, like a
   kernel launch, so time it with CUDA events.
5. **Check status codes**: every call returns a library-specific status
   (`cufftResult`, `cublasStatus_t`, ...). Solvers also report numerical
   failures through a device-side `devInfo` integer.
6. **Destroy** handles, plans and descriptors.

## cuFFT – `cufft_example.cu`

**Purpose.** 2D circular convolution through the frequency domain, using the
convolution theorem: `s ⊛ f = IFFT(FFT(s) · FFT(f))`.

**How it is written.**
- `cufftPlan2d(&plan, N, N, CUFFT_C2C)` builds one plan for an N×N
  complex-to-complex transform. The same plan serves the forward and inverse
  transforms of both arrays; direction is an argument to `cufftExecC2C`
  (`CUFFT_FORWARD` / `CUFFT_INVERSE`).
- The transforms are **in place** (input pointer == output pointer).
- `ComplexMUL` is a hand-written kernel for the pointwise product. cuFFT only
  does transforms, so any frequency-domain processing is your own kernel.
- `cufftComplex` is `float2` (`.x` = real, `.y` = imag).

**Problems in the original, and what changed.**
- *Multiply bug.* `a[i].x` is overwritten, then the new value is used to compute
  `a[i].y`. All-ones input has a purely real spectrum (only the DC bin is
  non-zero), which hides the bug. With random input the original kernel is off
  by about 10% (`cufft_example 8 random orig` → FAIL). Fix: read both operands
  into registers first.
- *Unnormalized inverse.* cuFFT's inverse is not divided by N², so the original
  prints 625 (= 25 × 25) rather than the true convolution of 25. The fixed
  kernel multiplies by `1/(N*N)`.
- *No bounds check, `<<<N, N>>>` launch.* That only works while N ≤ 1024 and
  the grid exactly covers N². Replaced with `ceil(N²/256)` blocks of 256 and
  `if (i < size)`.
- `delete result, fg, fig;` only deletes `result` (comma operator), and arrays
  need `delete[]`.
- The vendored `helper_cuda.h` no longer compiles on CUDA 13 (it uses the
  removed `cudaDeviceProp::clockRate`/`computeMode`), so local check macros
  replace it.

**Results / takeaways.**

| N | plan | 2 fwd FFTs | inverse |
|---|------|-----------|---------|
| 256 | 148 ms | 0.13 ms | 0.04 ms |
| 1024 | 241 ms | 0.60 ms | 0.29 ms |
| 1000 (2³·5³) | 156 ms | 0.75 ms | 0.36 ms |
| 1009 (prime) | 529 ms | 1.94 ms | 0.99 ms |
| 4096 | 107 ms | 11.4 ms | 5.7 ms (128 MB workspace) |

- **Plan creation costs 100–1000× a transform.** Create plans once and reuse
  them (cuFFT plans are the FFTW model).
- **Sizes with small prime factors are fast.** A prime N forces the slower
  Bluestein algorithm (about 2.6× slower FFT and 3.4× slower planning here).
  Pad to 2^a·3^b·5^c·7^d when you can.
- Large plans allocate a work area as big as the data (128 MB at 4096²).
  Check `cufftGetSize` on memory-limited GPUs.
- For real input, `CUFFT_R2C`/`C2R` would store only N×(N/2+1) outputs, about
  half the memory and work.
- FP32 FFT convolution matched the direct O(N⁴) computation to about 1e-7
  relative error.

## cuBLAS – `cublas_example.cu`

**Purpose.** Dense matrix multiply `C = α·A·B + β·C` (SGEMM) with
α = 1, β = 0.

**How it is written** (legacy `cublas.h` API, still present in CUDA 13):
- `cublasInit()` creates an implicit global context. `cublasAlloc` and
  `cublasFree` wrap `cudaMalloc` and `cudaFree`.
- `cublasSetMatrix(rows, cols, elemSize, hostA, lda, devA, ldda)` copies a
  matrix to the device, and `cublasGetMatrix` copies it back.
- `cublasSgemm('n','n', m, n, k, α, A, lda, B, ldb, β, C, ldc)`.
  `'n'` = no transpose, and `lda`/`ldb`/`ldc` are **leading dimensions** (rows
  of the stored column-major array).
- **Column-major storage.** `#define index(i,j,ld) ((j)*(ld)+(i))` is the
  Fortran/BLAS layout. Element (i, j) sits at `j*ld + i`. Passing C-style
  row-major arrays gives you the transpose. The usual trick is to compute
  `Cᵀ = Bᵀ·Aᵀ` by swapping the operands.
- The legacy API has no handle, so errors come from `cublasGetError()`. The
  modern `cublas_v2.h` API (used in the cuSPARSE/cuSOLVER examples) passes an
  explicit `cublasHandle_t` and returns a status from every call. The v2 SGEMM
  is `cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc)`,
  with α and β passed by pointer.

**What changed.** Runtime sizes `[HA WA WB]`, a CPU check, event timing of
upload / first call / steady state / download, and printing only for small
matrices. The ENTER prompt still appears when the program runs with no
arguments, as in the original.

**Results / takeaways.**

| Shape (m×k · k×n) | steady SGEMM | GFLOP/s | naive CPU |
|-------------------|-------------|---------|-----------|
| 2×9 · 9×2 (original) | 0.038 ms | ~0 | – |
| 512³ | 0.18 ms | 1489 | 455 ms |
| 1024³ | 1.38 ms | 1559 | 2650 ms (~1900× slower) |
| 4096³ | 102 ms | 1352 | – |
| 4096×16 · 16×4096 | 0.71 ms | 752 | – (but D2H of C took 20 ms) |
| 16×65536 · 65536×16 | 1.13 ms | 30 | – |

- **The first call is slow** (9–58 ms), from library and kernel loading.
  Warm up before timing anything.
- **Small problems don't benefit.** The 2×2 result spends more time in transfer
  and launch overhead than in arithmetic.
- **Shape matters as much as size.** A long inner dimension with a tiny output
  (16×65536·65536×16) has little parallelism or reuse and reaches only
  30 GFLOP/s. A large square GEMM reaches about 1.5 TFLOP/s (close to half this
  laptop GPU's FP32 peak).
- When the output is large, PCIe can dominate: the 4096×4096 C took 20 ms to
  download versus 0.7 ms to compute.

## cuSPARSE + cuBLAS – `cusparse_example.cu`

**Purpose.** Solve `A·x = b` for a large sparse symmetric positive-definite
(tridiagonal) matrix with the **Conjugate Gradient** method. This is NVIDIA's
`conjugateGradient` sample.

**How it is written.**
- `genTridiag` builds the matrix directly in **CSR** (compressed sparse row)
  form:
  - `I` (row offsets, N+1 entries): row r's non-zeros are `[I[r], I[r+1])`.
  - `J` (column index of each non-zero) and `val` (its value).
  - `val[start] = val[start-1]` copies the upper off-diagonal into the lower
    one, which makes A symmetric. A diagonal of about 10 against off-diagonals
    below 1 makes it strongly diagonally dominant, so it is positive definite
    and well conditioned.
- The CG loop:

  ```
  r = b - A·x;  p = r
  repeat: Ap = A·p                    ← cusparseSpMV  (the only sparse op)
          α = (r·r)/(p·Ap)            ← cublasSdot
          x += α·p;  r -= α·Ap        ← cublasSaxpy
          β = r_new·r_new / r·r;  p = r + β·p   ← cublasSscal + cublasSaxpy
  ```

  This shows how the libraries **compose**: cuSPARSE does the sparse
  matrix-vector product, cuBLAS level-1 does everything else, and all vectors
  stay on the GPU the whole time.
- **Ported to the generic API.** `cusparseScsrmv` was removed in CUDA 11. The
  matrix is now wrapped in a `cusparseSpMatDescr_t` (`cusparseCreateCsr`) and
  the vectors in `cusparseDnVecDescr_t` (`cusparseCreateDnVec`). One call to
  `cusparseSpMV_bufferSize` sizes a scratch buffer that every `cusparseSpMV`
  then reuses. Descriptors only *describe* existing device arrays; they don't
  copy them.
- Runtime arguments: `[N] [diag] [tol] [max_iter]`. `diag` controls the
  conditioning.

**Results / takeaways.**
- **Sparse storage is the whole point.** N = 1M needs 28 MB in CSR versus 4 TB
  dense. N = 16.7M needs 448 MB versus 1 EB.
- **CG iterations depend on conditioning, not size.** N = 1K, 64K, 1M and 16.7M
  took 6, 7, 8 and 8 iterations. Lowering the diagonal from 10 to 4, 2.5 and 2.1
  raised that to 8, 12, 18 and 22. A looser tolerance (1e-2) stopped at 12.
- **SpMV is memory-bound.** About 98 GB/s effective, which is near this GPU's
  DRAM bandwidth. At N = 1024 it reached only 1 GB/s because launch overhead
  dominates. GPUs need large problems.
- **Hidden synchronization.** An iteration took about 1.4 ms although its SpMV
  took 0.38 ms. `cublasSdot` with the default *host* pointer mode must return
  its scalar to the CPU, so every dot product stalls the pipeline (three per
  iteration, plus the explicit `cudaDeviceSynchronize`/`printf`).
  `cublasSetPointerMode(..., CUBLAS_POINTER_MODE_DEVICE)` keeps the scalars on
  the GPU.

## cuSOLVER + cuBLAS – `cusolver_example.cu`

**Purpose.** Solve a dense square system `A·x = b` by **QR factorization**,
which is LAPACK's `geqrf` + `ormqr` + `trsm` recipe.

**How it is written.**
1. Create a `cusolverDnHandle_t` (dense solver) and a `cublasHandle_t`.
2. Copy A (column-major, `lda = m`) and b to the device.
3. **Query workspace**: `cusolverDnDgeqrf_bufferSize` → `lwork`, then
   `cudaMalloc(d_work)`.
4. `cusolverDnDgeqrf`: in place, the upper triangle of `d_A` becomes **R**, and
   the Householder vectors below the diagonal, together with the scalars in
   `d_tau`, implicitly encode **Q**. `devInfo` reports success (0) or a bad
   argument (−i).
5. `cusolverDnDormqr(SIDE_LEFT, OP_T)` computes `b ← Qᵀ·b` without ever forming
   Q explicitly.
6. `cublasDtrsm(SIDE_LEFT, FILL_UPPER, OP_N, NON_UNIT)` solves the triangular
   system `R·x = Qᵀb` in place in `d_B`.
- cuSOLVER (LAPACK-style factorizations) and cuBLAS (BLAS triangular solve)
  share device data with no copies in between.

**Latent bug fixed.** The comment says "query working space of geqrf and
ormqr", but the original only queried `geqrf` and reused that `lwork` for
`ormqr`. Querying both shows `ormqr` needs **more** from m = 1024 up:

| m | geqrf lwork | ormqr lwork |
|---|-------------|-------------|
| 3 | 786,432 | 67,104 |
| 1024 | 1,572,864 | **2,425,632** |
| 4096 | 4,718,592 | **34,669,344** |

So the original code works for 3×3 only by luck. Always query every routine you
call and allocate the maximum.

**Results / takeaways.**
- Small systems are all overhead: about 9 ms total for 3×3, which a CPU solves
  in nanoseconds.
- geqrf speed: 1.9 GFLOP/s at m = 256, 22.5 at m = 1024, 85 at m = 4096. The
  GPU needs large matrices to fill up.
- 85 GFLOP/s is close to this GPU's **FP64** peak: consumer GPUs run double
  precision at 1/32 the FP32 rate, versus 1.35 TFLOP/s for FP32 SGEMM. Data
  center GPUs (A100/H100) have full-rate FP64.
- QR is backward stable: `max|x − 1|` stayed between 1e-15 and 5e-12, even for
  random 4096×4096 matrices. For SPD matrices, Cholesky (`potrf`/`potrs`) is
  about 2× cheaper, and LU (`getrf`/`getrs`) is the usual general choice.

## cuRAND – `curand_example.cu`

**Purpose.** Generate random numbers on the GPU with the cuRAND **device API**
(each thread owns a generator state).

**How it is written.**
- `curandState_t` is a XORWOW generator state, 48 bytes per thread.
- `curand_init(seed, subsequence, offset, &state)`:
  - `seed` selects the overall stream.
  - `subsequence = thread_idx` gives each thread its own non-overlapping
    2⁶⁷-long slice of that stream, so threads are statistically independent.
    Giving every thread the same subsequence makes them all produce the *same*
    numbers.
  - `offset` skips ahead within the subsequence.
- `curand(&state)` returns 32 random bits and advances the state.
  `% MAX` maps them to `[0, MAX)`. That has a slight modulo bias when MAX
  doesn't divide 2³², and `curand_uniform` is the cleaner choice for ranges.
- Three demonstrations:
  - `random0`: seed hard-coded to 1, so it prints **96 on every run**.
  - `random1`: seeded from `time()`, so it differs on each run.
  - `init` + `randoms`: one state per thread in global memory, a 25×20 grid of
    numbers.
- `random1(seed)` returns the same number as thread 0 of the grid with that
  seed: same seed, subsequence 0, offset 0, therefore the same first value.
- `#include <curand.h>` was included but the host API never used. Added at the
  end for comparison: `curandCreateGenerator` → `curandSetPseudoRandomGeneratorSeed`
  → `curandGenerate`, which fills a device buffer with no per-thread state to
  manage.

**What changed.** Runtime `[blocks] [threads_per_block] [max] [seed]`. The
output array moved from the stack (`unsigned int cpu_nums[N*M]` would overflow
for large grids) to the heap. Event timing and a mean/χ² uniformity check
were added. `<unistd.h>` (unused) is skipped on Windows.

**Results / takeaways.**

| numbers | device `init` | device generate | host API (2nd call) |
|---------|---------------|-----------------|---------------------|
| 500 | 0.12 ms | 0.26 ms | 0.04 ms |
| 1M | 37 ms | 0.8 ms | 0.05 ms |
| 4M | 211 ms | 3.1 ms | 0.15 ms |
| 16.7M | **842 ms** | 12.2 ms | 0.67 ms |

- **`curand_init` with distinct subsequences is expensive.** It runs a large
  skip-ahead per thread. Initialize states once and reuse them across kernel
  launches; never re-init per call. (Philox states are much cheaper to
  initialize.)
- One state per *output* doesn't scale. 16.7M × 48 B = 805 MB of state.
  Real code launches a modest grid and loops, so each thread generates many
  numbers from one state (kept in a register, written back at the end).
- **Use the host API for bulk numbers** (about 18× faster here). Use the device
  API when random numbers are consumed inside your own kernel (Monte Carlo,
  dropout, ...).
- **Reproducibility:** a fixed seed gives identical output across runs (the two
  `4 8 100 1234` runs match exactly), which is essential for debugging. Seed
  from time only when you need variety.
- All runs passed the χ² uniformity check.

## Environment notes (this machine)

- Base anaconda `nvcc` (CUDA 13.0) ships no math libraries. They came from a
  separate conda env:
  `conda create -n gpu_mod7 -c nvidia cuda-version=13.0 libcublas-dev libcufft-dev libcurand-dev libcusparse-dev libcusolver-dev`
- To run from Git Bash on Windows:

  ```bash
  E=$(cygpath -m ~/anaconda3_new/envs/gpu_mod7/Library)
  export PATH="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64:$HOME/anaconda3_new/envs/gpu_mod7/Library/bin:$PATH"
  NVCC_EXTRA="-I$E/include -L$E/lib/x64" ./run_practicals.sh
  ```

  On Linux with a normal CUDA toolkit, `./run_practicals.sh` alone works.
