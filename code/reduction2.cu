// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void
reductionCUDA(int *A, int *B)
{
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	sdata[tid] = A[i];
	__syncthreads();
	
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		B[blockIdx.x] = sdata[0];
	}
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
	

	// Allocate device memory
	int *d_A, *d_B;

	cudaError_t error;

	error = cudaMalloc((void **)&d_A, mem_size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, mem_size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	
	
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start2;
	error = cudaEventCreate(&start2);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start2 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop2;
	error = cudaEventCreate(&stop2);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop2 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start2, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start2 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	
	

	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start1;
	error = cudaEventCreate(&start1);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start1 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop1;
	error = cudaEventCreate(&stop1);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop1 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start1, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start1 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	
	// Setup execution parameters
	int num_threads = 64;
	dim3 threads(64, 1, 1);
	dim3 grid(262144, 1, 1);
	// Execute the kernel
	reductionCUDA << < grid, threads, num_threads * sizeof(int) >> > (d_A, d_B);
	
	// Setup execution parameters
	int num_threads2 = 64;
	dim3 threads2(64, 1, 1);
	dim3 grid2(4096, 1, 1);
	// Execute the kernel
	reductionCUDA << < grid2, threads2, num_threads2 * sizeof(int) >> > (d_B, d_A);
	
	// Setup execution parameters
	int num_threads3 = 64;
	dim3 threads3(64, 1, 1);
	dim3 grid3(64, 1, 1);
	// Execute the kernel
	reductionCUDA << < grid3, threads3, num_threads3 * sizeof(int) >> > (d_A, d_B);
	
	// Setup execution parameters
	int num_threads4 = 64;
	dim3 threads4(64, 1, 1);
	dim3 grid4(1, 1, 1);
	// Execute the kernel
	reductionCUDA << < grid4, threads4, num_threads4 * sizeof(int) >> > (d_B, d_A);
	

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	

	// Record the stop event
	error = cudaEventRecord(stop1, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop1 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop1);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop1 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal1 = 0.0f;
	error = cudaEventElapsedTime(&msecTotal1, start1, stop1);

	printf("Calculation elapsed time in msec = %f\n", msecTotal1);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	error = cudaMemcpy(h_B, d_A, 1, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_B,d_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
	
	
	// Record the stop event
	error = cudaEventRecord(stop2, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop2 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop2);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop2 event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal2 = 0.0f;
	error = cudaEventElapsedTime(&msecTotal2, start2, stop2);

	printf("Total elapsed time in msec = %f\n", msecTotal2);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	

	// Clean up memory
	free(h_A);
	free(h_B);
	cudaFree(d_A);
	cudaFree(d_B);

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

	printf("Array size is %d\n", n);

	int reduction_result = Reduction(argc, argv, n);

	exit(reduction_result);
}
