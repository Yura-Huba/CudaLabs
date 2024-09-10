
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#define _USE_MATH_DEFINES
#define ARRAY_SIZE 1000000000

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
 

typedef double type;

__global__ void calculateSin(type* ptrToArray, size_t arraySize) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < arraySize) {
		ptrToArray[index] = sinf(((type)(index % 360)) * M_PI / 180);
	}
}

double calculateSineError(type* ptrToArray, size_t arraySize) {
	double totalError = 0;
	for (int i = 0; i < arraySize; i++) {
		totalError += abs(sin((i % 360) * M_PI / 180) - ptrToArray[i]);
	}
	return totalError / arraySize;
}

int main() {
	const size_t arraySize = ARRAY_SIZE;
	type* ptrToArrayOnDevice;
	cudaError_t cudaError;

	int deviceCount = 0;
	cudaError = cudaGetDeviceCount(&deviceCount);

	if (cudaSuccess != cudaError) {
		printf("Description: %s\n", cudaGetErrorString(cudaError));
		return EXIT_FAILURE;
	}

	int device = 0;
	cudaSetDevice(device);
	
	cudaDeviceProp deviceProp;
	cudaError = cudaGetDeviceProperties(&deviceProp, device);

	if (cudaSuccess != cudaError) {
		printf("Description: %s\n", cudaGetErrorString(cudaError));
		return EXIT_FAILURE;
	}

	cudaError = cudaMalloc(&ptrToArrayOnDevice, sizeof(type) * arraySize);

	if (cudaSuccess != cudaError) {
		printf("Description: %s\n", cudaGetErrorString(cudaError));
		return EXIT_FAILURE;
	}

	dim3 threads = dim3(deviceProp.maxThreadsPerBlock);
	dim3 blocks = dim3((unsigned int) ceil((double) arraySize / deviceProp.maxThreadsPerBlock));

	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	calculateSin <<<blocks, threads >>> (ptrToArrayOnDevice, arraySize);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Time = %0.8f\n", elapsedTime);


	type* ptrToArrayOnHost;
	ptrToArrayOnHost = (type*)malloc(sizeof(type) * arraySize);

	if (ptrToArrayOnHost == NULL) {
		printf("No memory host\n");
		return EXIT_FAILURE;
	}

	cudaError = cudaMemcpy(ptrToArrayOnHost, ptrToArrayOnDevice, sizeof(type) * arraySize, cudaMemcpyDeviceToHost);

	if (cudaSuccess != cudaError) {
		printf("Description: %s\n", cudaGetErrorString(cudaError));
		return EXIT_FAILURE;
	}

	printf("Error = %0.10f \n", calculateSineError(ptrToArrayOnHost, arraySize));

	cudaFree(ptrToArrayOnDevice);
	free(ptrToArrayOnHost);

	return EXIT_SUCCESS;
}