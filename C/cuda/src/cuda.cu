#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void matrix_multiply_float(int n, float A[], float B[], float C[])
{
  cublasStatus_t status;
  cublasHandle_t handle;
  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;
  float alpha = 1.0;
  float beta = 0.0;
  cudaMalloc((void **)&d_A, n * n * sizeof(d_A[0]));
  cudaMalloc((void **)&d_B, n * n * sizeof(d_B[0]));
  cudaMalloc((void **)&d_C, n * n * sizeof(d_C[0]));

  cublasSetVector(n * n, sizeof(*A), A, 1, d_A, 1);
  cublasSetVector(n * n, sizeof(*B), B, 1, d_B, 1);
  cublasSetVector(n * n, sizeof(*C), C, 1, d_C, 1);
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return;
  }

  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A,
                       n, d_B, n, &beta, d_C, n);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! CUBLAS Sgemm error\n");
    return;
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return;
  }
}

extern "C" void matrix_multiply_double(int n, double A[], double B[], double C[])
{
  cublasStatus_t status;
  cublasHandle_t handle;
  double *d_A = 0;
  double *d_B = 0;
  double *d_C = 0;
  double alpha = 1.0;
  double beta = 0.0;
  cudaMalloc((void **)&d_A, n * n * sizeof(d_A[0]));
  cudaMalloc((void **)&d_B, n * n * sizeof(d_B[0]));
  cudaMalloc((void **)&d_C, n * n * sizeof(d_C[0]));

  cublasSetVector(n * n, sizeof(*A), A, 1, d_A, 1);
  cublasSetVector(n * n, sizeof(*B), B, 1, d_B, 1);
  cublasSetVector(n * n, sizeof(*C), C, 1, d_C, 1);
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return;
  }

  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A,
                       n, d_B, n, &beta, d_C, n);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! CUBLAS Dgemm error\n");
    return;
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return;
  }
}
