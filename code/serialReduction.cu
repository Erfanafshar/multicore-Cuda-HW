// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define TILE_WIDTH 16
void reductionSerial(int *A, int *B, int n)
{
	int res = 0;
	for (int i = 0; i < n; ++i)
	{
		res += A[i];
	}
	B[0] = res;
}

void randomInit(int *data, int size)
{
	srand(time(NULL));
	
	for (int i = 0; i < size; ++i)
	{
		data[i] = rand() % 10;
	}
}

/**
* Run a simple test of matrix multiplication using CUDA
*/
int Reduction(int argc, char **argv, int n)
{
	// Allocate host memory for array
	unsigned int size_A = n;
	unsigned int mem_size_A = sizeof(int)* size_A;
	int *h_A = (int *)malloc(mem_size_A);
	unsigned int size_B = n;
	unsigned int mem_size_B = sizeof(int)* size_B;
	int *h_B = (int *)malloc(mem_size_B);

	// Initialize Array
	randomInit(h_A, size_A);


	cudaError_t error;

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Execute the kernel
	reductionSerial(h_A, h_B, n);

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	printf("Elapsed time in msec = %f\n", msecTotal);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	


	// Clean up memory
	free(h_A);
	free(h_B);

	return EXIT_SUCCESS;
}


/**
* Program main
*/
int main(int argc, char **argv)
{
	printf("[Reduction Using CUDA] - Starting...\n");

	// By default, we use device 0
	int devID = 0;
	cudaSetDevice(devID);

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	// Size of square matrices
	size_t n = 0;
	printf("[-] N = ");
	scanf("%u", &n);

	printf("Array size is (%d, %d)\n", n, 1);

	int reduction_result = Reduction(argc, argv, n);

	exit(reduction_result);
}
