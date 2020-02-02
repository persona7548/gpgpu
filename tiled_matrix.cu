#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#define LEN_M 3*1024
#define LEN_N 2*1024
#define LEN_K 2*1024

#define TILE_WIDTH 16


__global__ void TiledMatrixMulKernel(int m, int n, int k, float* A, float* B, float* C)
{
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;

	float Cvalue = 0;
	for (int t = 0; t < n / TILE_WIDTH; ++t) {	
		
		ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH + tx];
		ds_B[ty][tx] = B[(t*TILE_WIDTH + ty)*k + Col];
		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; ++i)  // compute Cvalue based on ds_A[][] and ds_B[][]
			Cvalue += ds_A[ty][i] * ds_B[i][tx];
		__syncthreads();

	}
	C[Row*k + Col] = Cvalue;
}

int main()
{
	// Allocate and initialize the matrices A, B, C
	float * A, *B, *C;

	A = (float*)malloc(LEN_M*LEN_N * sizeof(float));
	B = (float*)malloc(LEN_N*LEN_K * sizeof(float));
	C = (float*)malloc(LEN_M*LEN_K * sizeof(float));

	for (int i = 0; i<LEN_M*LEN_N; i++) A[i] = i % 3;
	for (int i = 0; i<LEN_N*LEN_K; i++) B[i] = i % 4;

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

	TiledMatrixMulKernel << <dimGrid, dimBlock >> >(LEN_M, LEN_N, LEN_K, dev_A, dev_B, dev_C);
	cudaDeviceSynchronize();

	// I/O to write the output matrix C
	cudaMemcpy(C, dev_C, LEN_M*LEN_K * sizeof(float), cudaMemcpyDeviceToHost);



	// Free matrices A, B, C
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	free(A);
	free(B);
	free(C);

	return 0;
}
