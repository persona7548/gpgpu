#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 2048     // size of vectors
#define T 240       // number of threads per block

__global__ void vecAdd(int *A, int *B, int *C)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
		printf("%d\n",i);
		printf("%d\n", blockIdx.x);
	}
	
}

int main()
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// allocate the memory on the GPU
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	// fill the arrays 'a' and 'b' on the CPU
	for (int i = 0; i<N; i++) {
		a[i] = i;
		b[i] = i + i;
	}

	// copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice);

	int blocks = (N - 1) / T + 1;
	vecAdd << <blocks, T >> >(dev_a, dev_b, dev_c);

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost);

	// display the results
	long temp1, temp2;
	temp1 = temp2 = 0;
	for (int i = 0; i<N; i++) {
		temp1 += (a[i] + b[i]);
		temp2 += c[i];
	}
	printf("total a+b:%ld \n total c: %ld\n", temp1, temp2);

	// free the memory allocated on the GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}

