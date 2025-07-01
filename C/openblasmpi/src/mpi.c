#include <cblas.h>
#include <openblas_config.h>
#include <mpi.h>

// Parallel matrix multiply using OpenBLAS and MPI for float
void matrix_multiply_float(int n, int rank, int size, float local_A[], float B[], float local_C[])
{
	// 设置 OpenBLAS 使用单线程
	openblas_set_num_threads(1);

	// 计算每个进程处理的行数
	int rows_per_process = n / size;
	int remainder = n % size;
	if (rank < remainder)
	{
		rows_per_process++;
	}

	// 使用 OpenBLAS 进行局部矩阵乘法
	// C = α * A * B + β * C，这里 α = 1.0，β = 1.0
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				rows_per_process, n, n,
				1.0, local_A, n,
				B, n,
				1.0, local_C, n);
}

// Parallel matrix multiply using OpenBLAS and MPI for double
void matrix_multiply_double(int n, int rank, int size, double local_A[], double B[], double local_C[])
{
	// 设置 OpenBLAS 使用单线程
	openblas_set_num_threads(1);

	// 计算每个进程处理的行数
	int rows_per_process = n / size;
	int remainder = n % size;
	if (rank < remainder)
	{
		rows_per_process++;
	}

	// 使用 OpenBLAS 进行局部矩阵乘法
	// C = α * A * B + β * C，这里 α = 1.0，β = 1.0
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				rows_per_process, n, n,
				1.0, local_A, n,
				B, n,
				1.0, local_C, n);
}
