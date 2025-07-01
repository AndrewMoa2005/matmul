#include <cblas.h>

void matrix_multiply_float(int n, float A[], float B[], float C[])
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A, n, B, n, 0.0, C, n);
}

void matrix_multiply_double(int n, double A[], double B[], double C[])
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, A, n, B, n, 0.0, C, n);
}