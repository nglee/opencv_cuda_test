#include "cuda_runtime.h"
#include <nppi.h>
#include <cstdio>
#include <random>
#include <ctime>

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

constexpr size_t h = 987;
constexpr size_t w = 968;
constexpr size_t c = 3;
constexpr size_t size = h * w * c;

unsigned char h_src[size];
unsigned char h_dst[size];

int main()
{
    // Repeat test
    for (int cnt = 0; cnt < 100; ++cnt)
    {
        // Set input data with random value
        std::srand(std::time(nullptr));
        for (size_t i = 0; i < size; ++i)
            h_src[i] = std::rand() / ((RAND_MAX + 1) / 255);
        unsigned char* d_srcDst;
        size_t step = 0;

        checkCudaErrors(cudaDeviceReset());
        checkCudaErrors(cudaMallocPitch(&d_srcDst, &step, w * c, h));
        checkCudaErrors(cudaMemcpy2D(d_srcDst, step, h_src, w * c, w * c, h, cudaMemcpyHostToDevice));

        NppStatus status = nppiMirror_8u_C3IR(d_srcDst, step, NppiSize{ w, h }, NPP_BOTH_AXIS);
        //NppStatus status = nppiMirror_8u_C3IR(d_srcDst, step, NppiSize{ w, h }, NPP_HORIZONTAL_AXIS);
        //NppStatus status = nppiMirror_8u_C3IR(d_srcDst, step, NppiSize{ w, h }, NPP_VERTICAL_AXIS);
        if (status != NPP_NO_ERROR)
        {
            printf("npp error\n");
            return 1;
        }

        checkCudaErrors(cudaMemcpy2D(h_dst, w * c, d_srcDst, step, w * c, h, cudaMemcpyDeviceToHost));

        // Validate result
        for (size_t _h = 0; _h < h; ++_h)
        {
            for (size_t _w = 0; _w < w; ++_w)
            {
                for (size_t _c = 0; _c < c; ++_c)
                {
                    const unsigned char src_val = h_src[_h * w * c + _w * c + _c];
                    const unsigned char dst_val = h_dst[(h - 1 - _h) * w * c + (w - 1 - _w) * c + _c]; // both-axis flip
                    //const unsigned char dst_val = h_dst[(h - 1 - _h) * w * c + _w * c + _c]; // horizontal axis flip
                    //const unsigned char dst_val = h_dst[_h * w * c + (w - 1 - _w) * c + _c]; // vertical axis flip

                    if (src_val != dst_val)
                    {
                        printf("wrong!\n");
                        return 1;
                    }
                }
            }
        }

        printf("complete %d\n", cnt);

        checkCudaErrors(cudaFree(d_srcDst));
    }

    return 0;
}