#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N   (33 * 1024)

__global__ void add(int *a, int *b, int *c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
		printf("%d\n", tid);
	}
	
}

int main(void) {
	clock_t start, end;
	int *a, *b, *c, *d;
	int *dev_a, *dev_b, *dev_c;

	start = clock();
	// allocate the memory on the CPU
	a = (int*)malloc(N * sizeof(int));
	b = (int*)malloc(N * sizeof(int));
	c = (int*)malloc(N * sizeof(int));
	d = (int*)malloc(N * sizeof(int));

	// allocate the memory on the GPU
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	// fill the arrays 'a' and 'b' on the CPU
	for (int i = 0; i<N; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}

	// copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	add << <128, 128 >> >(dev_a, dev_b, dev_c);

	// copy the array 'c' back from the GPU to the CPU
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	printf(" kernel execution time: %lf sec\n", (clock_t)(end - start) / (float)CLOCKS_PER_SEC);
	// set d[i] by adding a[i] and b[i]. Compare d[i] and c[i]. If there is an error, print an error message.

	// free the memory we allocated on the GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	// free the memory we allocated on the CPU
	free(a);
	free(b);
	free(c);
	free(d);

	return 0;
}

