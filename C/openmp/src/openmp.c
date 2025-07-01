// Reference : https://zhuanlan.zhihu.com/p/667189867
#include <omp.h>

// Parallel matrix multiply
void matrix_multiply_float(int n, float A[], float B[], float C[])
{
#pragma omp parallel for collapse(2) shared(A, B, C)
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			C[i * n + j] = 0;
			for (int k = 0; k < n; k++)
			{
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}

void matrix_multiply_double(int n, double A[], double B[], double C[])
{
#pragma omp parallel for collapse(2) shared(A, B, C)
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			C[i * n + j] = 0;
			for (int k = 0; k < n; k++)
			{
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}
