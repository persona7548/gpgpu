
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define LEN_M 4
#define LEN_N 4
#define LEN_K 3

#define TILE_WIDTH 2

__global__ void MatrixMulOnDevice(int m, int n, int k, float * A, float * B, float * C) {
	int row = threadIdx.y + blockIdx.y*blockDim.y;
	int col = threadIdx.x + blockIdx.x*blockDim.x;

	if ((row<m) && (col<k)) {
		float temp = 0.0;
		for (int i = 0; i<n; i++) {
			temp += A[row*n + i] * B[i*k + col];
		}
		C[row*k + col] = temp;
	}
}

int main()
{
	// Allocate and initialize the matrices A, B, C
	float * A, *B, *C;

	A = (float*)malloc(LEN_M*LEN_N * sizeof(float));
	B = (float*)malloc(LEN_N*LEN_K * sizeof(float));
	C = (float*)malloc(LEN_M*LEN_K * sizeof(float));

	for (int i = 0; i<LEN_M*LEN_N; i++) A[i] = i;
	for (int i = 0; i<LEN_N*LEN_K; i++) B[i] = i;
	for (int i = 0; i<LEN_M*LEN_K; i++) C[i] = 0.0;

	// I/O to read the input matrices A and B
	float * dev_A, *dev_B, *dev_C;
	cudaMalloc((void**)&dev_A, LEN_M*LEN_N * sizeof(float));
	cudaMalloc((void**)&dev_B, LEN_N*LEN_K * sizeof(float));
	cudaMalloc((void**)&dev_C, LEN_M*LEN_K * sizeof(float));


	cudaMemcpy(dev_A, A, LEN_M*LEN_N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, LEN_N*LEN_K * sizeof(float), cudaMemcpyHostToDevice);

	// A*B on the device
	dim3 dimGrid((LEN_K - 1) / TILE_WIDTH + 1, (LEN_M - 1) / TILE_WIDTH + 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	MatrixMulOnDevice << <dimGrid, dimBlock >> >(LEN_M, LEN_N, LEN_K, dev_A, dev_B, dev_C);

	// I/O to write the output matrix C
	cudaMemcpy(C, dev_C, LEN_M*LEN_K * sizeof(float), cudaMemcpyDeviceToHost);

	printf("C:");
	for (int i = 0; i<LEN_M*LEN_K; i++) printf(" %6.1f", C[i]);
	printf("\n");

	// Free matrices A, B, C
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	free(A);
	free(B);
	free(C);
	return 0;
}

