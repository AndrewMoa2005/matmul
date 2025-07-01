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

	// 局部矩阵乘法
	for (int i = 0; i < rows_per_process; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
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

	// 局部矩阵乘法
	for (int i = 0; i < rows_per_process; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
			}
		}
	}
}