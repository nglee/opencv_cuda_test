#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <stdio.h>

void inline _CheckCudaError(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess)
        printf("%s - %s (%s:%d)\n", cudaGetErrorName(err), cudaGetErrorString(err), file, line);
}

#define CheckCudaError(call) _CheckCudaError((call), __FILE__, __LINE__)

__global__ void hello()
{
    printf("%u: hello\n", threadIdx.x);
}

using namespace cv::cuda;

int main()
{
    setBufferPoolUsage(true); // cuda-memcheck --leak-check full fails if this line exist, delete or set its argument to false to pass leak check test

    hello<<<1, 16, 0, StreamAccessor::getStream(Stream::Null())>>>();
    CheckCudaError(cudaDeviceSynchronize());

    CheckCudaError(cudaDeviceReset()); // For accurate leak checking (https://docs.nvidia.com/cuda/cuda-memcheck/index.html#leak-checking)
}