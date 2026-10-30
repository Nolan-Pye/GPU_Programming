
/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include ormqr_example.cpp
 *   nvcc -o -fopenmp a.out ormqr_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 *
 * Solves A*x = b with a QR factorization (A = Q*R):
 *   geqrf  -> A = Q*R            (cuSOLVER)
 *   ormqr  -> b' = Q^T * b       (cuSOLVER)
 *   trsm   -> x = R \ b'         (cuBLAS triangular solve)
 *
 * Usage: cusolver_example [m]
 *
 * With no arguments (or m = 3) it solves the original hard-coded 3x3 system
 * whose exact solution is x = (1 1 1)'. For m > 3 it builds a random m x m
 * matrix A, sets b = A * ones(m), and checks that x comes back as all ones.
 * Matrices are printed for m <= 4.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>


void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}


int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    const int m = (argc > 1) ? atoi(argv[1]) : 3;
    const int lda = m;
    const int ldb = m;
    const int nrhs = 1; // number of right hand side vectors
/*       | 1 2 3 |
 *   A = | 4 5 6 |
 *       | 2 1 1 |
 *
 *   x = (1 1 1)'
 *   b = (6 15 4)'
 */

    if (m < 1) {
        printf("Usage: %s [m]\n", argv[0]);
        return EXIT_FAILURE;
    }

    double *A = (double*)malloc(sizeof(double) * lda * m);
    double *A0 = (double*)malloc(sizeof(double) * lda * m);   // copy of A for the residual check
    double *B = (double*)malloc(sizeof(double) * ldb * nrhs);
    double *XC = (double*)malloc(sizeof(double) * ldb * nrhs); // solution matrix from GPU

    if (m == 3) {
        const double A3[9] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
//      double X[ldb*nrhs] = { 1.0, 1.0, 1.0}; // exact solution
        const double B3[3] = { 6.0, 15.0, 4.0};
        for (int i = 0; i < 9; i++) A[i] = A3[i];
        for (int i = 0; i < 3; i++) B[i] = B3[i];
    } else {
        // random A in [-1, 1), column-major; b = A * (1 1 ... 1)'
        srand(2024);
        for (int i = 0; i < lda * m; i++) A[i] = 2.0 * rand() / RAND_MAX - 1.0;
        for (int row = 0; row < m; row++) {
            double s = 0.0;
            for (int col = 0; col < m; col++) s += A[row + col*lda];
            B[row] = s;
        }
    }
    for (int i = 0; i < lda * m; i++) A0[i] = A[i];

    double *d_A = NULL; // linear memory of GPU
    double *d_tau = NULL; // linear memory of GPU
    double *d_B  = NULL;
    int *devInfo = NULL; // info in gpu (device copy)
    double *d_work = NULL;
    int  lwork = 0;
    int  lwork_geqrf = 0;
    int  lwork_ormqr = 0;

    int info_gpu = 0;

    const double one = 1;

    printf("Solving A*x = b for a %d x %d system (%s)\n\n", m, m,
           m == 3 ? "original hard-coded A and b" : "random A, b = A*ones");
    if (m <= 4) {
        printf("A = (matlab base-1)\n");
        printMatrix(m, m, A, lda, "A");
        printf("=====\n");
        printf("B = (matlab base-1)\n");
        printMatrix(m, nrhs, B, ldb, "B");
        printf("=====\n");
    }

    cudaEvent_t ev[6];
    for (int e = 0; e < 6; e++) cudaEventCreate(&ev[e]);

// step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&d_B  , sizeof(double) * ldb * nrhs);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m   , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * ldb * nrhs, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);


// step 3: query working space of geqrf and ormqr
//         (the original only asked geqrf and reused that size for ormqr;
//          ask both and allocate the larger so neither call can run short)
    cusolver_status = cusolverDnDgeqrf_bufferSize(
        cusolverH,
        m,
        m,
        d_A,
        lda,
        &lwork_geqrf);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusolver_status = cusolverDnDormqr_bufferSize(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        m,
        nrhs,
        m,
        d_A,
        lda,
        d_tau,
        d_B,
        ldb,
        &lwork_ormqr);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    lwork = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;
    printf("workspace: geqrf %d, ormqr %d doubles -> allocating %d\n", lwork_geqrf, lwork_ormqr, lwork);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

// step 4: compute QR factorization
    cudaEventRecord(ev[0]);
    cusolver_status = cusolverDnDgeqrf(
        cusolverH,
        m,
        m,
        d_A,
        lda,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cudaEventRecord(ev[1]);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after geqrf: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

// step 5: compute Q^T*B
    cudaEventRecord(ev[2]);
    cusolver_status= cusolverDnDormqr(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        m,
        nrhs,
        m,
        d_A,
        lda,
        d_tau,
        d_B,
        ldb,
        d_work,
        lwork,
        devInfo);
    cudaEventRecord(ev[3]);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);


    // check if QR is good or not
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("after ormqr: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

// step 6: compute x = R \ Q^T*B

    cudaEventRecord(ev[4]);
    cublas_status = cublasDtrsm(
         cublasH,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         m,
         nrhs,
         &one,
         d_A,
         lda,
         d_B,
         ldb);
    cudaEventRecord(ev[5]);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(XC, d_B, sizeof(double)*ldb*nrhs, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    if (m <= 4) {
        printf("X = (matlab base-1)\n");
        printMatrix(m, nrhs, XC, ldb, "X");
    }

    // Accuracy: distance from the exact solution (all ones) and the residual
    double maxXErr = 0.0, maxRes = 0.0;
    for (int row = 0; row < m; row++) {
        double ax = 0.0;
        for (int col = 0; col < m; col++) ax += A0[row + col*lda] * XC[col];
        maxRes = fmax(maxRes, fabs(ax - B[row]));
        maxXErr = fmax(maxXErr, fabs(XC[row] - 1.0));
    }

    float geqrfMs, ormqrMs, trsmMs;
    cudaEventElapsedTime(&geqrfMs, ev[0], ev[1]);
    cudaEventElapsedTime(&ormqrMs, ev[2], ev[3]);
    cudaEventElapsedTime(&trsmMs, ev[4], ev[5]);
    double geqrfFlops = 4.0 / 3.0 * (double)m * m * m;  // Householder QR of a square matrix

    printf("\nTimings (FP64):\n");
    printf("  geqrf (A = QR)     : %9.3f ms  (%.2f GFLOP/s)\n", geqrfMs, geqrfFlops / (geqrfMs * 1e6));
    printf("  ormqr (Q^T * b)    : %9.3f ms\n", ormqrMs);
    printf("  trsm  (R \\ Q^T b)  : %9.3f ms\n", trsmMs);
    printf("max |x - 1| = %.3e, max |A*x - b| = %.3e -> %s\n",
           maxXErr, maxRes, maxXErr < 1e-6 ? "PASS" : "FAIL");

// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_tau  ) cudaFree(d_tau);
    if (d_B    ) cudaFree(d_B);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    for (int e = 0; e < 6; e++) cudaEventDestroy(ev[e]);
    free(A); free(A0); free(B); free(XC);


    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);

    cudaDeviceReset();

    return maxXErr < 1e-6 ? 0 : 1;
}
