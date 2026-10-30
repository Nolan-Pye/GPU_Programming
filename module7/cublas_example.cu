/*
 * C = A * B with cuBLAS SGEMM (legacy cublas.h API), column-major storage.
 *
 * Usage: cublas_example [HA WA WB] [-noprompt]
 *
 * With no arguments it runs the original configuration: A is 2x9, B is 9x2,
 * C is 2x2, and it waits for ENTER before exiting (pass -noprompt to skip).
 *
 *   HA - rows of A (and C)
 *   WA - columns of A = rows of B (the inner dimension k)
 *   WB - columns of B (and C)
 *
 * Matrices are printed when they are at most 10x10. The result is checked
 * against a CPU GEMM (all of C when it is small enough, otherwise a sample),
 * and the upload, first SGEMM call, steady-state SGEMM and download are timed.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas.h>
#define HA 2
#define WA 9
#define WB 2
#define HB WA
#define WC WB
#define HC HA
#define index(i,j,ld) (((j)*(ld))+(i))

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); } } while (0)

void printMat(float*P,int uWP,int uHP){
  //printf("\n %f",P[1]);
  int i,j;
  for(i=0;i<uHP;i++){

      printf("\n");

      for(j=0;j<uWP;j++)
          printf("%f ",P[index(i,j,uHP)]);
  }
}

/* One element of C computed on the CPU in double precision */
double cpuElement(const float *A, const float *B, int ha, int wa, int hb, int i, int j){
  double sum = 0.0;
  for (int k = 0; k < wa; k++)
    sum += (double)A[index(i,k,ha)] * B[index(k,j,hb)];
  return sum;
}


 int  main (int argc, char** argv) {
    cublasStatus status;
    int i,j;
    int ha = HA, wa = WA, wb = WB;
    int npos = 0;
    /* any argument means "no prompt" (as in the original); numbers set sizes */
    for (int a = 1; a < argc; a++) {
      if (!strcmp(argv[a], "-noprompt") || !strcmp(argv[a], "-qatest")) continue;
      else if (npos == 0) { ha = atoi(argv[a]); npos++; }
      else if (npos == 1) { wa = atoi(argv[a]); npos++; }
      else if (npos == 2) { wb = atoi(argv[a]); npos++; }
    }
    int hb = wa, wc = wb, hc = ha;
    if (ha < 1 || wa < 1 || wb < 1) {
      fprintf (stderr, "Usage: %s [HA WA WB] [-noprompt]\n", argv[0]);
      return EXIT_FAILURE;
    }
    printf("Config: A is %dx%d, B is %dx%d, C is %dx%d\n", ha, wa, hb, wb, hc, wc);

    cublasInit();

    float *A = (float*)malloc((size_t)ha*wa*sizeof(float));
    float *B = (float*)malloc((size_t)hb*wb*sizeof(float));
    float *C = (float*)malloc((size_t)hc*wc*sizeof(float));
    if (A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    if (B == 0) {
        fprintf (stderr, "!!!! host memory allocation error (B)\n");
        return EXIT_FAILURE;
    }
    if (C == 0) {
      fprintf (stderr, "!!!! host memory allocation error (C)\n");
      return EXIT_FAILURE;
    }


    for (i=0;i<ha;i++)
      for (j=0;j<wa;j++)
        A[index(i,j,ha)] = (float) index(i,j,ha);
    for (i=0;i<hb;i++)
      for (j=0;j<wb;j++)
        B[index(i,j,hb)] = (float) index(i,j,hb);
    /*
    for (i=0;i<HA*WA;i++)
    A[i]=(float) i;
    for (i=0;i<HB*WB;i++)
    B[i]=(float) i;         */


        float* AA; float* BB; float* CC;

    cudaEvent_t ev[6];
    for (int e = 0; e < 6; e++) CUDA_CHECK(cudaEventCreate(&ev[e]));

    /*ALLOCATE ON THE DEVICE*/
    status=cublasAlloc(ha*wa,sizeof(float),(void**)&AA);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    status=cublasAlloc(hb*wb,sizeof(float),(void**)&BB);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (B)\n");
      return EXIT_FAILURE;
    }

    status=cublasAlloc(hc*wc,sizeof(float),(void**)&CC);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (C)\n");
      return EXIT_FAILURE;
    }

    /*SET MATRIX*/
    CUDA_CHECK(cudaEventRecord(ev[0]));
    status=cublasSetMatrix(ha,wa,sizeof(float),A,ha,AA,ha);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory write error (A)\n");
      return EXIT_FAILURE;
    }

    status=cublasSetMatrix(hb,wb,sizeof(float),B,hb,BB,hb);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory write error (B)\n");
      return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaEventRecord(ev[1]));

    /*KERNEL*/
    cublasSgemm('n','n',ha,wb,wa,1,AA,ha,BB,hb,0,CC,hc);
    CUDA_CHECK(cudaEventRecord(ev[2]));

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
    }

    /* Repeat the same SGEMM to measure steady-state speed (the first call
       above includes one-time library/kernel loading). */
    const int reps = 10;
    CUDA_CHECK(cudaEventRecord(ev[3]));
    for (int r = 0; r < reps; r++)
      cublasSgemm('n','n',ha,wb,wa,1,AA,ha,BB,hb,0,CC,hc);
    CUDA_CHECK(cudaEventRecord(ev[4]));

    status = cublasGetMatrix(hc,wc,sizeof(float),CC,hc,C,hc);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device read error (C)\n");
      return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaEventRecord(ev[5]));
    CUDA_CHECK(cudaEventSynchronize(ev[5]));


    /* PERFORMANCE OUTPUT*/

    if (ha <= 10 && wa <= 10 && wb <= 10) {
      printf("\nMatriz A:\n");
      printMat(A,wa,ha);
      printf("\nMatriz B:\n");
      printMat(B,wb,hb);
      printf("\nMatriz C:\n");
      printMat(C,wc,hc);
      printf("\n");
    }

    float setMs, firstMs, loopMs, getMs;
    CUDA_CHECK(cudaEventElapsedTime(&setMs, ev[0], ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&firstMs, ev[1], ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&loopMs, ev[3], ev[4]));
    CUDA_CHECK(cudaEventElapsedTime(&getMs, ev[4], ev[5]));
    double flops = 2.0 * ha * wb * wa;
    double sgemmMs = loopMs / reps;

    /* CPU check: every element if the whole product is cheap, else a sample */
    long long cElems = (long long)hc * wc;
    bool full = flops <= 2.2e9;
    int samples = full ? (int)cElems : 1024;
    double maxRel = 0.0;
    srand(42);
    auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < samples; s++) {
      long long idx = full ? s : (((long long)rand() * RAND_MAX + rand()) % cElems);
      int ci = (int)(idx % hc), cj = (int)(idx / hc);
      double ref = cpuElement(A, B, ha, wa, hb, ci, cj);
      double rel = fabs(C[index(ci,cj,hc)] - ref) / (fabs(ref) + 1.0);
      if (rel > maxRel) maxRel = rel;
    }
    double cpuMs = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - cpuStart).count();

    printf("\nTimings:\n");
    printf("  cublasSetMatrix A,B (H2D)   : %9.3f ms\n", setMs);
    printf("  first cublasSgemm           : %9.3f ms\n", firstMs);
    printf("  steady-state cublasSgemm    : %9.3f ms  (%.2f GFLOP/s, avg of %d)\n",
           sgemmMs, flops / (sgemmMs * 1e6), reps);
    printf("  cublasGetMatrix C (D2H)     : %9.3f ms\n", getMs);
    if (full)
      printf("  CPU reference, all of C     : %9.3f ms  (%.2f GFLOP/s, naive loop in double)\n",
             cpuMs, flops / (cpuMs * 1e6));
    printf("Check vs CPU (%s, %d element%s): max rel error = %.3e -> %s\n",
           full ? "all of C" : "random sample", samples, samples == 1 ? "" : "s",
           maxRel, maxRel < 1e-4 ? "PASS" : "FAIL");

    for (int e = 0; e < 6; e++) cudaEventDestroy(ev[e]);
    free( A );  free( B );  free ( C );
    status = cublasFree(AA);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (A)\n");
      return EXIT_FAILURE;
    }
    status = cublasFree(BB);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (B)\n");
      return EXIT_FAILURE;
    }
    status = cublasFree(CC);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (C)\n");
      return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! shutdown error (A)\n");
      return EXIT_FAILURE;
    }

    if (argc == 1) {
      printf("\nPress ENTER to exit...\n");
      getchar();
    }

		return maxRel < 1e-4 ? EXIT_SUCCESS : EXIT_FAILURE;


  }
