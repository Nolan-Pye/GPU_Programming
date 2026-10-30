/*
 * Module 8 practical: nvgraph_Pagerank.cpp ported to cuSPARSE + Thrust.
 *
 * nvGRAPH was deprecated in CUDA 10 and removed in CUDA 11, so the original
 * example cannot be built with a current toolkit. This file reproduces what
 * nvgraphPagerank() does on the same graph, in the same CSC layout, with the
 * same two runs (alpha = 0.85 from a uniform start, then alpha = 0.90 starting
 * from the first result as the initial guess).
 *
 * PageRank is the dominant eigenvector of the Google matrix
 *     G = alpha * (H + a * 1^T / n) + (1 - alpha) * 1 * 1^T / n
 * (H = weighted transition matrix, a = bookmark of dangling vertices), found
 * with the power method:
 *     x_{k+1} = alpha * H x_k + (alpha * a.x_k + (1 - alpha) * sum(x_k)) / n
 *     x_{k+1} /= ||x_{k+1}||_1,   stop when ||x_{k+1} - x_k||_1 < tol
 *
 * The CSC "destination_offsets / source_indices" arrays of nvGRAPH are exactly
 * CSR of H with rows = destination vertices, so H x is one cusparseSpMV call.
 *
 * Usage (cuda-samples style flags, all optional):
 *   pagerank_cusparse                       original 6-vertex graph
 *   pagerank_cusparse -alpha=0.5 -tol=1e-3  change damping / tolerance
 *   pagerank_cusparse -n=1000000 -deg=8     random graph, n vertices, avg out-degree deg
 *   (also -seed=S, -maxiter=M)
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <helper_string.h>
#include <helper_cuda.h>
#include <cusparse.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

#define CHECK_CUSPARSE(call)                                                  \
    do {                                                                      \
        cusparseStatus_t s_ = (call);                                         \
        if (s_ != CUSPARSE_STATUS_SUCCESS) {                                  \
            printf("cuSPARSE error %s at %s:%d\n", cusparseGetErrorString(s_), \
                   __FILE__, __LINE__);                                       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Graph in nvGRAPH's CSC-32 layout (host side).
struct CscGraph
{
    int n = 0, nnz = 0;
    std::vector<int> destination_offsets;   // n + 1
    std::vector<int> source_indices;        // nnz
    std::vector<float> weights;             // nnz, 1/outdegree(source)
    std::vector<float> bookmark;            // n, 1 if dangling
};

// The 6-vertex example from nvgraph_Pagerank.cpp, values copied verbatim.
CscGraph sampleGraph()
{
    CscGraph g;
    g.n = 6; g.nnz = 10;
    g.weights = {0.333333f, 0.500000f, 0.333333f, 0.500000f, 0.500000f,
                 1.000000f, 0.333333f, 0.500000f, 0.500000f, 0.500000f};
    g.destination_offsets = {0, 1, 3, 4, 6, 8, 10};
    g.source_indices = {2, 0, 2, 0, 4, 5, 2, 3, 3, 4};
    g.bookmark = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    return g;
}

// Random directed graph: each vertex gets an out-degree in [0, 2*deg]
// (degree 0 makes it dangling); edge targets are skewed toward low vertex ids
// so a few vertices collect most links and clearly "win" PageRank.
CscGraph randomGraph(int n, int deg, unsigned int seed)
{
    CscGraph g;
    g.n = n;
    srand(seed);
    auto urand = []() { return (rand() * (RAND_MAX + 1.0) + rand()) / ((RAND_MAX + 1.0) * (RAND_MAX + 1.0)); };

    std::vector<int> src, dst;
    std::vector<int> outdeg(n, 0);
    src.reserve((size_t)n * deg);
    dst.reserve((size_t)n * deg);
    for (int v = 0; v < n; ++v)
    {
        int d = (int)(urand() * (2 * deg + 1));
        for (int e = 0; e < d; ++e)
        {
            double u = urand();
            int t = (int)(n * u * u * u);
            if (t == v) t = (t + 1) % n;
            src.push_back(v);
            dst.push_back(t);
        }
        outdeg[v] = d;
    }
    g.nnz = (int)src.size();

    // counting sort edges by destination -> CSC
    g.destination_offsets.assign(n + 1, 0);
    for (int t : dst) g.destination_offsets[t + 1]++;
    for (int v = 0; v < n; ++v) g.destination_offsets[v + 1] += g.destination_offsets[v];
    std::vector<int> pos(g.destination_offsets.begin(), g.destination_offsets.end() - 1);
    g.source_indices.resize(g.nnz);
    g.weights.resize(g.nnz);
    for (int e = 0; e < g.nnz; ++e)
    {
        int p = pos[dst[e]]++;
        g.source_indices[p] = src[e];
        g.weights[p] = 1.0f / outdeg[src[e]];
    }
    g.bookmark.resize(n);
    for (int v = 0; v < n; ++v) g.bookmark[v] = outdeg[v] == 0 ? 1.0f : 0.0f;
    return g;
}

struct abs_diff
{
    __host__ __device__ float operator()(const thrust::tuple<float, float> &t) const
    {
        return fabsf(thrust::get<0>(t) - thrust::get<1>(t));
    }
};

struct axpb
{
    float s;
    __host__ __device__ axpb(float s_) : s(s_) {}
    __host__ __device__ float operator()(float y) const { return y + s; }
};

struct scale_by
{
    float s;
    __host__ __device__ scale_by(float s_) : s(s_) {}
    __host__ __device__ float operator()(float y) const { return y * s; }
};

// GPU PageRank. Mirrors nvgraphPagerank(handle, graph, weight, alpha,
// bookmark, has_guess, pagerank, tolerance, max_iter).
class GpuPagerank
{
public:
    explicit GpuPagerank(const CscGraph &g)
        : n(g.n), nnz(g.nnz),
          offsets(g.destination_offsets.begin(), g.destination_offsets.end()),
          indices(g.source_indices.begin(), g.source_indices.end()),
          values(g.weights.begin(), g.weights.end()),
          bookmark(g.bookmark.begin(), g.bookmark.end()),
          x(n), y(n)
    {
        CHECK_CUSPARSE(cusparseCreate(&handle));
        CHECK_CUSPARSE(cusparseCreateCsr(&matH, n, n, nnz,
                                         thrust::raw_pointer_cast(offsets.data()),
                                         thrust::raw_pointer_cast(indices.data()),
                                         thrust::raw_pointer_cast(values.data()),
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, thrust::raw_pointer_cast(x.data()), CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n, thrust::raw_pointer_cast(y.data()), CUDA_R_32F));
        float one = 1.0f, zero = 0.0f;
        size_t bytes = 0;
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matH,
                                               vecX, &zero, vecY, CUDA_R_32F,
                                               CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
        buffer.resize(bytes > 0 ? bytes : 1);
    }

    ~GpuPagerank()
    {
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroySpMat(matH);
        cusparseDestroy(handle);
    }

    // Returns the iteration count; result left in pr.
    int run(float alpha, const std::vector<float> *guess, float tol, int maxIter,
            std::vector<float> &pr, float &ms, float &spmvMs, float &residual)
    {
        cudaEvent_t start, stop, s0, s1;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventCreate(&s0);
        cudaEventCreate(&s1);
        cudaEventRecord(start);
        spmvMs = 0.0f;

        if (guess)
        {
            thrust::copy(guess->begin(), guess->end(), x.begin());
        }
        else
        {
            thrust::fill(x.begin(), x.end(), 1.0f / n);
        }

        float zero = 0.0f;
        int it = 0;
        residual = 0.0f;
        for (it = 1; it <= maxIter; ++it)
        {
            // y = alpha * H x   (x and y are swapped each iteration, see below)
            cudaEventRecord(s0);
            CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matH,
                                        vecX, &zero, vecY, CUDA_R_32F,
                                        CUSPARSE_SPMV_ALG_DEFAULT,
                                        thrust::raw_pointer_cast(buffer.data())));
            cudaEventRecord(s1);
            // rank leaking from dangling vertices and teleportation, spread evenly
            float dangling = thrust::inner_product(bookmark.begin(), bookmark.end(), x.begin(), 0.0f);
            float total = thrust::reduce(x.begin(), x.end(), 0.0f);
            float t;   // inner_product returned to the host, so s1 has completed
            cudaEventElapsedTime(&t, s0, s1);
            spmvMs += t;
            float shift = (alpha * dangling + (1.0f - alpha) * total) / n;
            thrust::transform(y.begin(), y.end(), y.begin(), axpb(shift));
            // L1-normalize, then measure the change
            float norm = thrust::reduce(y.begin(), y.end(), 0.0f);
            thrust::transform(y.begin(), y.end(), y.begin(), scale_by(1.0f / norm));
            residual = thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(y.begin(), x.begin())),
                thrust::make_zip_iterator(thrust::make_tuple(y.end(), x.end())),
                abs_diff(), 0.0f, thrust::plus<float>());
            x.swap(y);
            CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, thrust::raw_pointer_cast(x.data())));
            CHECK_CUSPARSE(cusparseDnVecSetValues(vecY, thrust::raw_pointer_cast(y.data())));
            if (residual < tol) break;
        }
        if (it > maxIter) it = maxIter;

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(s0);
        cudaEventDestroy(s1);

        pr.resize(n);
        thrust::copy(x.begin(), x.end(), pr.begin());
        return it;
    }

private:
    int n, nnz;
    thrust::device_vector<int> offsets, indices;
    thrust::device_vector<float> values, bookmark, x, y;
    thrust::device_vector<char> buffer;
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matH;
    cusparseDnVecDescr_t vecX, vecY;
};

// Same algorithm on the CPU (double precision) as a reference.
int cpuPagerank(const CscGraph &g, double alpha, double tol, int maxIter, std::vector<double> &x)
{
    int n = g.n;
    x.assign(n, 1.0 / n);
    std::vector<double> y(n);
    int it;
    for (it = 1; it <= maxIter; ++it)
    {
        double dangling = 0, total = 0;
        for (int v = 0; v < n; ++v) { dangling += g.bookmark[v] * x[v]; total += x[v]; }
        double shift = (alpha * dangling + (1 - alpha) * total) / n;
        double norm = 0;
        for (int d = 0; d < n; ++d)
        {
            double s = 0;
            for (int e = g.destination_offsets[d]; e < g.destination_offsets[d + 1]; ++e)
                s += g.weights[e] * x[g.source_indices[e]];
            y[d] = alpha * s + shift;
            norm += y[d];
        }
        double res = 0;
        for (int v = 0; v < n; ++v) { y[v] /= norm; res += fabs(y[v] - x[v]); }
        x.swap(y);
        if (res < tol) break;
    }
    return it > maxIter ? maxIter : it;
}

void printTop(const std::vector<float> &pr, int k)
{
    std::vector<int> idx(pr.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    k = std::min<int>(k, (int)idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return pr[a] > pr[b]; });
    printf("top %d vertices:", k);
    for (int i = 0; i < k; ++i) printf("  %d (%.5f)", idx[i], pr[idx[i]]);
    printf("\n");
}

int main(int argc, char **argv)
{
    const char **cargv = (const char **)argv;
    int cuda_device = findCudaDevice(argc, cargv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    float alpha1 = 0.85f, alpha2 = 0.90f;
    float tol = 1e-6f;
    int maxIter = 500;
    int n = 0, deg = 8;
    unsigned int seed = 1234;
    if (checkCmdLineFlag(argc, cargv, "alpha")) alpha1 = alpha2 = getCmdLineArgumentFloat(argc, cargv, "alpha");
    if (checkCmdLineFlag(argc, cargv, "tol")) tol = getCmdLineArgumentFloat(argc, cargv, "tol");
    if (checkCmdLineFlag(argc, cargv, "maxiter")) maxIter = getCmdLineArgumentInt(argc, cargv, "maxiter");
    if (checkCmdLineFlag(argc, cargv, "n")) n = getCmdLineArgumentInt(argc, cargv, "n");
    if (checkCmdLineFlag(argc, cargv, "deg")) deg = getCmdLineArgumentInt(argc, cargv, "deg");
    if (checkCmdLineFlag(argc, cargv, "seed")) seed = getCmdLineArgumentInt(argc, cargv, "seed");

    bool sample = (n == 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    CscGraph g = sample ? sampleGraph() : randomGraph(n, deg, seed);
    double buildMs = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    int dangling = 0;
    for (float b : g.bookmark) dangling += b != 0.0f;
    printf("graph: %d vertices, %d edges, %d dangling%s\n", g.n, g.nnz, dangling,
           sample ? " (nvgraph_Pagerank.cpp sample)" : "");
    if (!sample) printf("  host graph build: %.1f ms\n", buildMs);
    printf("tolerance %.1e (L1 change), max %d iterations\n\n", tol, maxIter);

    t0 = std::chrono::high_resolution_clock::now();
    GpuPagerank solver(g);
    double setupMs = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    std::vector<float> pr_1, pr_2;
    float ms1, ms2, spmv1, spmv2, res1, res2;

    // First run with default values (uniform start)
    int it1 = solver.run(alpha1, nullptr, tol, maxIter, pr_1, ms1, spmv1, res1);
    printf("pr_1, alpha = %.2f  (%d iterations, %.3f ms, %.3f ms/iter, residual %.2e)\n",
           alpha1, it1, ms1, ms1 / it1, res1);
    printf("  SpMV share of solve time: %.0f%% (%.3f ms per SpMV)\n", 100.0f * spmv1 / ms1, spmv1 / it1);
    if (sample) { for (int i = 0; i < g.n; i++) printf("%f\n", pr_1[i]); printf("\n"); }
    else printTop(pr_1, 5);

    // Second run with a different damping factor and pr_1 as initial guess
    int it2 = solver.run(alpha2, &pr_1, tol, maxIter, pr_2, ms2, spmv2, res2);
    printf("pr_2, alpha = %.2f, initial guess = pr_1  (%d iterations, %.3f ms, residual %.2e)\n",
           alpha2, it2, ms2, res2);
    if (sample) { for (int i = 0; i < g.n; i++) printf("%f\n", pr_2[i]); printf("\n"); }
    else printTop(pr_2, 5);

    if (sample && alpha2 == 0.90f)
    {
        const float expected[6] = {0.037210f, 0.053960f, 0.041510f, 0.37510f, 0.206000f, 0.28620f};
        float maxErr = 0;
        for (int i = 0; i < 6; ++i) maxErr = std::max(maxErr, fabsf(pr_2[i] - expected[i]));
        printf("max |pr_2 - expected (Langville & Meyer)| = %.2e\n", maxErr);
    }

    // CPU reference for the first run
    std::vector<double> ref;
    t0 = std::chrono::high_resolution_clock::now();
    int itc = cpuPagerank(g, alpha1, tol, maxIter, ref);
    double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    double l1 = 0;
    for (int i = 0; i < g.n; ++i) l1 += fabs(pr_1[i] - ref[i]);
    printf("\nCPU reference (alpha %.2f, double): %d iterations, %.3f ms, %.3f ms/iter\n",
           alpha1, itc, cpuMs, cpuMs / itc);
    printf("  ||pr_1 - cpu||_1 = %.2e, GPU setup (H->D + cuSPARSE descriptors) %.1f ms\n", l1, setupMs);
    printf("  speedup per iteration: %.3gx\n", (cpuMs / itc) / (ms1 / it1));

    printf("\nDone!\n");
    return EXIT_SUCCESS;
}
