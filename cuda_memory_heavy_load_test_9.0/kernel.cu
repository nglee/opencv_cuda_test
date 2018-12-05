#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thread>

__global__ void kernel(unsigned char* devPtr)
{
    devPtr[threadIdx.x + blockDim.x * blockIdx.x] = (unsigned char)sqrtf(threadIdx.x);
}

void func()
{
    while (1) {
        unsigned char* devPtr;
        cudaError_t cudaStatus = cudaMalloc(&devPtr, 1024 * 1024 * 1024);
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "(cudaMalloc) %s : %s\n", cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus));
            return;
        }
        else {
            fprintf(stdout, "(cudaMalloc) success\n");
        }

        kernel<<<40, 512>>>(devPtr);
        cudaDeviceSynchronize();

        unsigned char* hPtr = (unsigned char*)malloc(1024 * 1024 * 1024);
        cudaMemcpy(hPtr, devPtr, 1024 * 1024 * 1024, cudaMemcpyDeviceToHost);

        cudaStatus = cudaFree(devPtr);
        if (cudaStatus != cudaSuccess) {
            fprintf(stdout, "(cudaFree) %s : %s\n", cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus));
            return;
        }
        else {
            fprintf(stdout, "(cudaFree) success\n");
        }

        free(hPtr);
    }
}

int main()
{
    std::thread t1(func);
    std::thread t2(func);
    std::thread t3(func);
    std::thread t4(func);

    while (1) {}
}