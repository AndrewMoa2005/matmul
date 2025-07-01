#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

extern void matrix_multiply_float(int n, float A[], float B[], float C[]);
extern void matrix_multiply_double(int n, double A[], double B[], double C[]);

// Initialize matrix
void initialize_matrix_float(int n, float matrix[])
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix[i * n + j] = rand() / (float)(RAND_MAX);
		}
	}
}

void initialize_matrix_double(int n, double matrix[])
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix[i * n + j] = rand() / (double)(RAND_MAX);
		}
	}
}

// Execute matrix multiply and print results
int execute_float(int dim, int loop_num, double *ave_gflops, double *max_gflops, double *ave_time, double *min_time)
{
	// Use volatile to prevent compiler optimizations
	volatile float *a, *b, *c;
	struct timespec start_ns, end_ns;
	double cpu_time;

	for (int i = 0; i < loop_num; i++)
	{ // Allocate memory for matrices
		a = (float *)malloc(dim * dim * sizeof(float));
		b = (float *)malloc(dim * dim * sizeof(float));
		c = (float *)malloc(dim * dim * sizeof(float));
		if (a == NULL || b == NULL || c == NULL)
		{
			fprintf(stderr, "Memory allocation failed\n");
			return 0;
		}

		initialize_matrix_float(dim, a);
		initialize_matrix_float(dim, b);

		timespec_get(&start_ns, TIME_UTC);
		matrix_multiply_float(dim, a, b, c);
		timespec_get(&end_ns, TIME_UTC);
		cpu_time = (end_ns.tv_sec - start_ns.tv_sec) + (end_ns.tv_nsec - start_ns.tv_nsec) / 1e9;
		double gflops = 1e-9 * dim * dim * dim * 2 / cpu_time;
		printf("%d\t: %d x %d Matrix multiply wall time : %.6fs(%.3fGflops)\n", i + 1, dim, dim, cpu_time, gflops);

		// Free memory
		free(a);
		free(b);
		free(c);
		*ave_gflops += gflops;
		*max_gflops = MAX(*max_gflops, gflops);
		*ave_time += cpu_time;
		*min_time = MIN(*min_time, cpu_time);
	}
	*ave_gflops /= loop_num;
	*ave_time /= loop_num;
	return 1;
}

int execute_double(int dim, int loop_num, double *ave_gflops, double *max_gflops, double *ave_time, double *min_time)
{
	// Use volatile to prevent compiler optimizations
	volatile double *a, *b, *c;
	struct timespec start_ns, end_ns;
	double cpu_time;

	for (int i = 0; i < loop_num; i++)
	{ // Allocate memory for matrices
		a = (double *)malloc(dim * dim * sizeof(double));
		b = (double *)malloc(dim * dim * sizeof(double));
		c = (double *)malloc(dim * dim * sizeof(double));
		if (a == NULL || b == NULL || c == NULL)
		{
			fprintf(stderr, "Memory allocation failed\n");
			return 0;
		}

		initialize_matrix_double(dim, a);
		initialize_matrix_double(dim, b);

		timespec_get(&start_ns, TIME_UTC);
		matrix_multiply_double(dim, a, b, c);
		timespec_get(&end_ns, TIME_UTC);
		cpu_time = (end_ns.tv_sec - start_ns.tv_sec) + (end_ns.tv_nsec - start_ns.tv_nsec) / 1e9;
		double gflops = 1e-9 * dim * dim * dim * 2 / cpu_time;
		printf("%d\t: %d x %d Matrix multiply wall time : %.6fs(%.3fGflops)\n", i + 1, dim, dim, cpu_time, gflops);

		// Free memory
		free(a);
		free(b);
		free(c);
		*ave_gflops += gflops;
		*max_gflops = MAX(*max_gflops, gflops);
		*ave_time += cpu_time;
		*min_time = MIN(*min_time, cpu_time);
	}
	*ave_gflops /= loop_num;
	*ave_time /= loop_num;
	return 1;
}

int main(int argc, char *argv[])
{
	int n = 10;								   // Default matrix size exponent
	int loop_num = 5;						   // Number of iterations for averaging
	double ave_gflops = 0.0, max_gflops = 0.0; // Average and maximum Gflops
	double ave_time = 0.0, min_time = 1e9;	   // Average and minimum time
	int use_double = 0;						   // Default to float precision

	// Help message
	if (argc == 1 || (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)))
	{
		printf("Usage: %s [-n SIZE] [-l LOOP_NUM] [-float|-double]\n", argv[0]);
		printf("  -n SIZE      Specify matrix size, like 2^SIZE (default: 10)\n");
		printf("  -l LOOP_NUM  Specify number of iterations (default: 5)\n");
		printf("  -float       Use float precision (default)\n");
		printf("  -double      Use double precision\n");
		printf("  -h, --help   Show this help message\n");
		return 0;
	}

	// Parse -n, -l, -float, -double options
	int double_flag = 0, float_flag = 0;
	for (int argi = 1; argi < argc; ++argi)
	{
		if (strcmp(argv[argi], "-n") == 0 && argi + 1 < argc)
		{
			n = atoi(argv[argi + 1]);
			argi++;
		}
		else if (strcmp(argv[argi], "-l") == 0 && argi + 1 < argc)
		{
			loop_num = atoi(argv[argi + 1]);
			argi++;
		}
		else if (strcmp(argv[argi], "-double") == 0)
		{
			double_flag = 1;
		}
		else if (strcmp(argv[argi], "-float") == 0)
		{
			float_flag = 1;
		}
	}
	if (double_flag && float_flag)
	{
		fprintf(stderr, "Error: Cannot specify both -double and -float options.\n");
		return 1;
	}
	use_double = double_flag ? 1 : 0;

	int dim = (int)pow(2, n);

	if (use_double)
	{
		printf("Using double precision for matrix multiplication.\n");
		execute_double(dim, loop_num, &ave_gflops, &max_gflops, &ave_time, &min_time);
	}
	else
	{
		printf("Using float precision for matrix multiplication.\n");
		execute_float(dim, loop_num, &ave_gflops, &max_gflops, &ave_time, &min_time);
	}
	printf("Average Gflops: %.3f, Max Gflops: %.3f\n", ave_gflops, max_gflops);
	printf("Average Time: %.6fs, Min Time: %.6fs\n", ave_time, min_time);

	return 0;
}
