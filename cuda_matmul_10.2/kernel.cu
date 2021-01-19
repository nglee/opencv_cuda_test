
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <cstdlib>

template <typename T>
void _check(T result, char const* const func, char const* const file, int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) _check((val), #val, __FILE__, __LINE__)

__global__
void matmul(const float* A, const float* B, float* C)
{
    
}

#define MATRIX_WIDTH (1ULL << 14)
#define MATRIX_HEIGHT (1ULL << 14)
#define MATRIX_SIZE (MATRIX_HEIGHT * MATRIX_WIDTH)
#define MALLOC_SIZE (sizeof(float) * MATRIX_SIZE)

int main()
{
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    // Allocate memory
    float* h_A = reinterpret_cast<float*>(std::malloc(MALLOC_SIZE));
    float* h_B = reinterpret_cast<float*>(std::malloc(MALLOC_SIZE));
    float* h_C = reinterpret_cast<float*>(std::malloc(MALLOC_SIZE));
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, MALLOC_SIZE));
    checkCudaErrors(cudaMalloc(&d_B, MALLOC_SIZE));
    checkCudaErrors(cudaMalloc(&d_C, MALLOC_SIZE));

    // Init memory
    for (int idx = 0; idx < MATRIX_SIZE; idx++) {
        h_A[idx] = dist(e2);
        h_B[idx] = dist(e2);
    }

    // Copy host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, MALLOC_SIZE, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, MALLOC_SIZE, cudaMemcpyHostToDevice));

    dim3 gridSize(1024);
    dim3 blockSize(1024);

    matmul << <gridSize, blockSize >> > (d_A, d_B, d_C);

    // Copy device to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, MALLOC_SIZE, cudaMemcpyDeviceToHost));

    // Free memory
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_A));
    free(h_C);
    free(h_B);
    free(h_A);
}

