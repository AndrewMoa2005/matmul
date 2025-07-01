// Reference : https://github.com/CNugteren/CLBlast/blob/master/samples/sgemm.c
#include <stdio.h>
// Includes the CLBlast library (C interface)
#include <clblast_c.h>

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// parallel Matrix Multiply
void matrix_multiply_float(int n, float A[], float B[], float C[])
{
	// OpenCL platform/device settings
	const size_t platform_id = 0;
	const size_t device_id = 0;
	// SGEMM/DGEMM arguments
	const float alpha = 0.7f;
	const float beta = 1.0f;

	// Initializes the OpenCL platform
	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id *platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
	clGetPlatformIDs(num_platforms, platforms, NULL);
	cl_platform_id platform = platforms[platform_id];

	// Initializes the OpenCL device
	cl_uint num_devices;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	cl_device_id device = devices[device_id];

	// Creates the OpenCL context, queue, and an event
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
	cl_event event = NULL;

	// Copy the matrices to the device
	cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(float), NULL, NULL);
	cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(float), NULL, NULL);
	cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, n * n * sizeof(float), A, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, n * n * sizeof(float), B, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, n * n * sizeof(float), C, 0, NULL, NULL);

	CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor,
											CLBlastTransposeNo, CLBlastTransposeNo,
											n, n, n,
											alpha,
											device_a, 0, n,
											device_b, 0, n,
											beta,
											device_c, 0, n,
											&queue, &event);
	// Wait for completion
	if (status == CLBlastSuccess)
	{
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
	}
	else
	{
		fprintf(stderr, "Error! CLBlast failed with status %d\n", status);
	}

	// Clean-up
	free(platforms);
	free(devices);
	clReleaseMemObject(device_a);
	clReleaseMemObject(device_b);
	clReleaseMemObject(device_c);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

void matrix_multiply_double(int n, double A[], double B[], double C[])
{
	// OpenCL platform/device settings
	const size_t platform_id = 0;
	const size_t device_id = 0;
	// SGEMM/DGEMM arguments
	const double alpha = 0.7f;
	const double beta = 1.0f;

	// Initializes the OpenCL platform
	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id *platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
	clGetPlatformIDs(num_platforms, platforms, NULL);
	cl_platform_id platform = platforms[platform_id];

	// Initializes the OpenCL device
	cl_uint num_devices;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	cl_device_id device = devices[device_id];

	// Creates the OpenCL context, queue, and an event
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
	cl_event event = NULL;

	// Copy the matrices to the device
	cl_mem device_a = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(double), NULL, NULL);
	cl_mem device_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(double), NULL, NULL);
	cl_mem device_c = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(double), NULL, NULL);
	clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, n * n * sizeof(double), A, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, n * n * sizeof(double), B, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, device_c, CL_TRUE, 0, n * n * sizeof(double), C, 0, NULL, NULL);

	CLBlastStatusCode status = CLBlastDgemm(CLBlastLayoutRowMajor,
											CLBlastTransposeNo, CLBlastTransposeNo,
											n, n, n,
											alpha,
											device_a, 0, n,
											device_b, 0, n,
											beta,
											device_c, 0, n,
											&queue, &event);

	// Wait for completion
	if (status == CLBlastSuccess)
	{
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
	}
	else
	{
		fprintf(stderr, "Error! CLBlast failed with status %d\n", status);
	}

	// Clean-up
	free(platforms);
	free(devices);
	clReleaseMemObject(device_a);
	clReleaseMemObject(device_b);
	clReleaseMemObject(device_c);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}