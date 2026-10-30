/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 * Usage: cusparse_example [N] [diag] [tol] [max_iter]
 *
 * With no arguments it runs the original configuration: N = 1048576,
 * diagonal entries in [10, 11), tol = 1e-5, max_iter = 10000.
 *
 *   N        - matrix is N x N (symmetric tridiagonal, stored as CSR), N >= 2
 *   diag     - diagonal entries are rand in [diag, diag+1); off-diagonals are
 *              rand in [0, 1). Smaller diag -> worse conditioned -> more
 *              CG iterations (keep diag > 2 so the matrix stays positive
 *              definite)
 *   tol      - stop when ||r|| < tol
 *   max_iter - iteration limit
 *
 * Ported to the CUDA 11+ generic cuSPARSE API: the legacy cusparseScsrmv()
 * used originally was removed in CUDA 11, so A*p is now cusparseSpMV() on a
 * CSR matrix descriptor and dense-vector descriptors.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

const char *sSDKname     = "conjugateGradient";

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

#define CUBLAS_CHECK(call) do { cublasStatus_t s_ = (call); if (s_ != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

#define CUSPARSE_CHECK(call) do { cusparseStatus_t s_ = (call); if (s_ != CUSPARSE_STATUS_SUCCESS) { \
        printf("cuSPARSE error %s at %s:%d\n", cusparseGetErrorString(s_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz, float diag)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + diag;
    val[1] = (float)rand()/RAND_MAX;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }

        val[start] = val[start-1];
        val[start+1] = (float)rand()/RAND_MAX + diag;

        if (i < N-1)
        {
            val[start+2] = (float)rand()/RAND_MAX;
        }
    }

    I[N] = nz;
}

int main(int argc, char **argv)
{
    int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    float *val = NULL;
    float tol = 1e-5f;
    int max_iter = 10000;
    float diag = 10.0f;
    float *x;
    float *rhs;
    float a, b, na, r0, r1;
    int *d_col, *d_row;
    float *d_val, *d_x, dot;
    float *d_r, *d_p, *d_Ax;
    int k;
    float alpha, beta, alpham1;

    N = (argc > 1) ? atoi(argv[1]) : 1048576;
    if (argc > 2) diag = (float)atof(argv[2]);
    if (argc > 3) tol = (float)atof(argv[3]);
    if (argc > 4) max_iter = atoi(argv[4]);
    if (N < 2)
    {
        printf("Usage: %s [N>=2] [diag] [tol] [max_iter]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Use the current (default) CUDA device
    cudaDeviceProp deviceProp;
    int devID = 0;
    CUDA_CHECK(cudaGetDevice(&devID));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, devID));

    // Statistics about the GPU device
    printf("%s: GPU Device %d: \"%s\"\n", sSDKname, devID, deviceProp.name);
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    M = N;
    nz = (N-2)*3 + 4;
    I = (int *)malloc(sizeof(int)*(N+1));
    J = (int *)malloc(sizeof(int)*nz);
    val = (float *)malloc(sizeof(float)*nz);
    genTridiag(I, J, val, N, nz, diag);

    double csrMB = ((N + 1) * sizeof(int) + nz * (sizeof(int) + sizeof(float))) / 1048576.0;
    double denseMB = (double)M * N * sizeof(float) / 1048576.0;
    printf("Matrix: %d x %d, nnz = %d, diag in [%g, %g), tol = %g, max_iter = %d\n",
           M, N, nz, diag, diag + 1.0f, tol, max_iter);
    printf("Storage: CSR %.2f MB vs dense %.2f MB (%.0fx smaller)\n\n",
           csrMB, denseMB, denseMB / csrMB);

    x = (float *)malloc(sizeof(float)*N);
    rhs = (float *)malloc(sizeof(float)*N);

    for (int i = 0; i < N; i++)
    {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    CUDA_CHECK(cudaMalloc((void **)&d_col, nz*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_val, nz*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_x, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_r, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_p, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_Ax, N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r, rhs, N*sizeof(float), cudaMemcpyHostToDevice));

    /* Wrap the raw device arrays in generic-API descriptors: one CSR sparse
       matrix (row offsets, column indices, values) and the dense vectors that
       appear in A*x and A*p. The descriptors don't copy data, they just
       describe it. */
    cusparseSpMatDescr_t matA = NULL;
    cusparseDnVecDescr_t vecx = NULL, vecp = NULL, vecAx = NULL;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_32F));

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    /* SpMV may need scratch space; ask how much, allocate it once, reuse it */
    size_t bufferSize = 0;
    void *d_buffer = NULL;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecx, &beta, vecAx, CUDA_R_32F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize > 0 ? bufferSize : 1));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    /* r = b - A*x0 */
    CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
                                &beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

    CUBLAS_CHECK(cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1));
    CUBLAS_CHECK(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

    k = 1;

    while (r1 > tol*tol && k <= max_iter)
    {
        if (k > 1)
        {
            b = r1 / r0;
            CUBLAS_CHECK(cublasSscal(cublasHandle, N, &b, d_p, 1));
            CUBLAS_CHECK(cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1));
        }
        else
        {
            CUBLAS_CHECK(cublasScopy(cublasHandle, N, d_r, 1, d_p, 1));
        }

        CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
                                    &beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
        CUBLAS_CHECK(cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot));
        a = r1 / dot;

        CUBLAS_CHECK(cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1));
        na = -a;
        CUBLAS_CHECK(cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1));

        r0 = r1;
        CUBLAS_CHECK(cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (k <= 10 || k % 25 == 0 || r1 <= tol*tol)
            printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float solveMs;
    CUDA_CHECK(cudaEventElapsedTime(&solveMs, start, stop));

    /* Time SpMV on its own to see how close it gets to memory bandwidth */
    const int reps = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++)
        CUSPARSE_CHECK(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
                                    &beta, vecAx, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float spmvMs;
    CUDA_CHECK(cudaEventElapsedTime(&spmvMs, start, stop));
    spmvMs /= reps;
    /* bytes touched: CSR arrays + read p + write Ax */
    double spmvBytes = (N + 1) * 4.0 + nz * 8.0 + N * 8.0;

    CUDA_CHECK(cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost));

    float rsum, diff, err = 0.0;

    for (int i = 0; i < N; i++)
    {
        rsum = 0.0;

        for (int j = I[i]; j < I[i+1]; j++)
        {
            rsum += val[j]*x[J[j]];
        }

        diff = fabs(rsum - rhs[i]);

        if (diff > err)
        {
            err = diff;
        }
    }

    int iters = k - 1;
    printf("\nCG %s after %d iteration%s, final residual = %e\n",
           (r1 <= tol*tol) ? "converged" : "did NOT converge", iters, iters == 1 ? "" : "s", sqrt(r1));
    printf("Timings:\n");
    printf("  CG solve (incl. initial residual) : %9.3f ms  (%.3f ms/iteration)\n",
           solveMs, iters > 0 ? solveMs / iters : solveMs);
    printf("  one cusparseSpMV                  : %9.3f ms  (~%.1f GB/s effective)\n",
           spmvMs, spmvBytes / (spmvMs * 1e6));

    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecx));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecp));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecAx));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    cudaFree(d_buffer);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    printf("Test Summary:  Error amount = %f\n", err);
    exit((k <= max_iter) ? 0 : 1);
}
