#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cstdio>

void check(cublasStatus_t ret)
{
	if (ret != CUBLAS_STATUS_SUCCESS)
		printf("cublas error : %d", ret);
}

void print_memory()
{
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	printf("%llu\n", free);
}

int main()
{
	int loop = 10000;
	while (loop--)
	{
		#pragma omp parallel num_threads(4)
		{
			cublasHandle_t handle;
			cudaStream_t stream;

			cudaSetDevice(0);
			cudaStreamCreate(&stream);
			check(cublasCreate(&handle));
			check(cublasSetStream(handle, stream));

			//print_memory();

			check(cublasDestroy(handle));
			cudaStreamDestroy(stream);

			//print_memory();
		}
	}
}