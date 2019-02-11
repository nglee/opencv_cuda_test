#include <stdio.h>

__global__
void kernel1()
{
	int tid = threadIdx.x;
	int val = tid;
	val = __shfl_up_sync(0xFFFFFFFFU, val, 1);
	printf("thread %i 's val is %i\n", tid, val); 
}

__global__
void kernel2()
{
	int tid = threadIdx.x;
	int val = tid;
	if (tid < 5)
		val = __shfl_up_sync(0xFFFFFFFFU, val, 1);
	printf("thread %i 's val is %i\n", tid, val); 
}

int main()
{
	kernel1<<<1,7>>>();
	cudaDeviceSynchronize();
	printf("kernel1 done\n");

	kernel2<<<1,7>>>();
	cudaDeviceSynchronize();
	printf("kernel2 done\n");
}
