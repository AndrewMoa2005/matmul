#include <math.h>

// 分块大小，可根据缓存大小调整
#define BLOCK_SIZE 8

// Parallel matrix multiply
void matrix_multiply_float(int n, int rank, int size, float local_A[], float B[], float local_C[])
{
    // 计算每个进程处理的行数
    int rows_per_process = n / size;
    int remainder = n % size;
    if (rank < remainder)
    {
        rows_per_process++;
    }

    // 初始化 local_C 为 0
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < n; j++) {
            local_C[i * n + j] = 0.0f;
        }
    }

    // 分块矩阵乘法
    for (int bi = 0; bi < rows_per_process; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
                // 块内计算
                int end_i = fmin(bi + BLOCK_SIZE, rows_per_process);
                int end_j = fmin(bj + BLOCK_SIZE, n);
                int end_k = fmin(bk + BLOCK_SIZE, n);
                for (int i = bi; i < end_i; i++) {
                    for (int j = bj; j < end_j; j++) {
                        float sum = local_C[i * n + j];
                        for (int k = bk; k < end_k; k++) {
                            sum += local_A[i * n + k] * B[k * n + j];
                        }
                        local_C[i * n + j] = sum;
                    }
                }
            }
        }
    }
}

// Parallel matrix multiply
void matrix_multiply_double(int n, int rank, int size, double local_A[], double B[], double local_C[])
{
    // 计算每个进程处理的行数
    int rows_per_process = n / size;
    int remainder = n % size;
    if (rank < remainder)
    {
        rows_per_process++;
    }

    // 初始化 local_C 为 0
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < n; j++) {
            local_C[i * n + j] = 0.0;
        }
    }

    // 分块矩阵乘法
    for (int bi = 0; bi < rows_per_process; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < n; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < n; bk += BLOCK_SIZE) {
                // 块内计算
                int end_i = fmin(bi + BLOCK_SIZE, rows_per_process);
                int end_j = fmin(bj + BLOCK_SIZE, n);
                int end_k = fmin(bk + BLOCK_SIZE, n);
                for (int i = bi; i < end_i; i++) {
                    for (int j = bj; j < end_j; j++) {
                        double sum = local_C[i * n + j];
                        for (int k = bk; k < end_k; k++) {
                            sum += local_A[i * n + k] * B[k * n + j];
                        }
                        local_C[i * n + j] = sum;
                    }
                }
            }
        }
    }
}