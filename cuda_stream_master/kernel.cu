
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <stdio.h>

void _CheckCudaError(const cudaError_t error, const char *func, const int line)
{
	if (error != cudaSuccess) {
		printf("%s : %s @%s(%d)\n", cudaGetErrorName(error), cudaGetErrorString(error), func, line);
		exit(EXIT_FAILURE);
	}
}

#define CheckCudaError(run) _CheckCudaError((run), __FUNCTION__, __LINE__)

__global__ void hello()
{
	printf("%u: hello\n", threadIdx.x);
}

int main()
{
	hello<<<1, 16, 0, cv::cuda::StreamAccessor::getStream(cv::cuda::Stream::Null())>>>();
	CheckCudaError(cudaDeviceSynchronize());
	CheckCudaError(cudaDeviceReset());
}