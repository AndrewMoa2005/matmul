// References: https://github.com/ROCm/rocBLAS-Examples/blob/develop/Languages/C/main.c

#include <assert.h>
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <rocblas/rocblas.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if (error != hipSuccess)                      \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
    }
#endif

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                  \
    if (status != rocblas_status_success)             \
    {                                                 \
        fprintf(stderr, "rocBLAS error: ");           \
        fprintf(stderr,                               \
                "rocBLAS error: '%s'(%d) at %s:%d\n", \
                rocblas_status_to_string(status),     \
                status,                               \
                __FILE__,                             \
                __LINE__);                            \
    }
#endif

void matrix_multiply_float(int n, float A[], float B[], float C[])
{
    size_t rows, cols;
    rows = cols = n;

    typedef float data_type;

    rocblas_handle handle;
    rocblas_status rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    hipStream_t test_stream;
    rstatus = rocblas_get_stream(handle, &test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);

    data_type *da = 0;
    data_type *db = 0;
    data_type *dc = 0;
    CHECK_HIP_ERROR(hipMalloc((void **)&da, n * cols * sizeof(data_type)));
    CHECK_HIP_ERROR(hipMalloc((void **)&db, n * cols * sizeof(data_type)));
    CHECK_HIP_ERROR(hipMalloc((void **)&dc, n * cols * sizeof(data_type)));

    // upload asynchronously from pinned memory
    rstatus = rocblas_set_matrix_async(rows, cols, sizeof(data_type), A, n, da, n, test_stream);
    rstatus = rocblas_set_matrix_async(rows, cols, sizeof(data_type), B, n, db, n, test_stream);

    // scalar arguments will be from host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    data_type alpha = 1.0;
    data_type beta = 2.0;

    // invoke asynchronous computation
    rstatus = rocblas_sgemm(handle,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            rows,
                            cols,
                            n,
                            &alpha,
                            da,
                            n,
                            db,
                            n,
                            &beta,
                            dc,
                            n);
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch results asynchronously to pinned memory
    rstatus = rocblas_get_matrix_async(rows, cols, sizeof(data_type), dc, n, C, n, test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);

    // wait on transfer to be finished
    CHECK_HIP_ERROR(hipStreamSynchronize(test_stream));

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);
}

void matrix_multiply_double(int n, double A[], double B[], double C[])
{
    size_t rows, cols;
    rows = cols = n;

    typedef double data_type;

    rocblas_handle handle;
    rocblas_status rstatus = rocblas_create_handle(&handle);
    CHECK_ROCBLAS_STATUS(rstatus);

    hipStream_t test_stream;
    rstatus = rocblas_get_stream(handle, &test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);

    data_type *da = 0;
    data_type *db = 0;
    data_type *dc = 0;
    CHECK_HIP_ERROR(hipMalloc((void **)&da, n * cols * sizeof(data_type)));
    CHECK_HIP_ERROR(hipMalloc((void **)&db, n * cols * sizeof(data_type)));
    CHECK_HIP_ERROR(hipMalloc((void **)&dc, n * cols * sizeof(data_type)));

    // upload asynchronously from pinned memory
    rstatus = rocblas_set_matrix_async(rows, cols, sizeof(data_type), A, n, da, n, test_stream);
    rstatus = rocblas_set_matrix_async(rows, cols, sizeof(data_type), B, n, db, n, test_stream);

    // scalar arguments will be from host memory
    rstatus = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    CHECK_ROCBLAS_STATUS(rstatus);

    data_type alpha = 1.0;
    data_type beta = 2.0;

    // invoke asynchronous computation
    rstatus = rocblas_dgemm(handle,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            rows,
                            cols,
                            n,
                            &alpha,
                            da,
                            n,
                            db,
                            n,
                            &beta,
                            dc,
                            n);
    CHECK_ROCBLAS_STATUS(rstatus);

    // fetch results asynchronously to pinned memory
    rstatus = rocblas_get_matrix_async(rows, cols, sizeof(data_type), dc, n, C, n, test_stream);
    CHECK_ROCBLAS_STATUS(rstatus);

    // wait on transfer to be finished
    CHECK_HIP_ERROR(hipStreamSynchronize(test_stream));

    CHECK_HIP_ERROR(hipFree(da));
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));

    rstatus = rocblas_destroy_handle(handle);
    CHECK_ROCBLAS_STATUS(rstatus);
}