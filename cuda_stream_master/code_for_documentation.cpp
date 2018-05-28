#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
//
//#include <iostream>
//#include <chrono>

using namespace cv;
using namespace cv::cuda;
//using namespace std::chrono;
//
//std::pair<size_t, size_t> stackSizePresets[7] = { { 0x000100000, /*   1MB */ 512},
//                                                  { 0x000400000, /*   4MB */ 1024},
//                                                  { 0x001000000, /*  16MB */ 2048},
//                                                  { 0x004000000, /*  64MB */ 4096},
//                                                  { 0x010000000, /* 256MB */ 8192},
//                                                  { 0x040000000, /*   1GB */ 16384},
//                                                  { 0x100000000, /*   4GB */ 32768},
//                                                };
//
//#define NUM_STREAM              5
//#define USE_BUFFER_POOL         true
//#define STACK_SIZE_PRESET_IDX   0    
//
//#define FOR_EACH_STREAM(i)  for (int i = 0; i < NUM_STREAM; i++)
//
//__global__ void
//dummy_kernel(void)
//{
//    printf("Hello\n");
//}

//int main()
//{
//    cudaDeviceReset();
//    cudaDeviceSynchronize();
//
//    auto tick = high_resolution_clock::now();
//
//    const size_t stackSize = stackSizePresets[STACK_SIZE_PRESET_IDX].first;
//    const size_t size = stackSizePresets[STACK_SIZE_PRESET_IDX].second;
//    const Size matSize(size, size);
//
//    setBufferPoolUsage(USE_BUFFER_POOL);
//    if (USE_BUFFER_POOL)
//        setBufferPoolConfig(getDevice(), stackSize, NUM_STREAM); // 64 MB per stream
//
//    Stream stream[NUM_STREAM];
//    BufferPool *pool[NUM_STREAM];
//    GpuMat d_src[NUM_STREAM];
//    GpuMat d_dst[NUM_STREAM];
//
//    FOR_EACH_STREAM(i) {
//        pool[i] = new BufferPool(stream[i]);
//    }
//
//    FOR_EACH_STREAM(i) {
//        d_src[i] = pool[i]->getBuffer(matSize, CV_8UC1); // 16 MB
//        d_dst[i] = pool[i]->getBuffer(matSize, CV_8UC3); // 48 MB
//    }
//
//    float wallClockTime = duration<float, std::milli>(high_resolution_clock::now() - tick).count();
//    std::cout << wallClockTime << " ms\n";
//
//    for (int i = 0; i < NUM_STREAM; i++)
//        delete pool[i];
//}

//int main()
//{
//    cudaDeviceSynchronize();
//
//    setBufferPoolUsage(true);                               // Tell OpenCV that we are going to utilize StackAllocator
//    setBufferPoolConfig(getDevice(), 1024 * 1024 * 64, 2);  // Allocate 64 MB, 2 stacks (default is 10 MB, 5 stacks)
//
//    Stream stream1, stream2;                                // Each stream uses 1 stack
//    BufferPool pool1(stream1), pool2(stream2);
//
//    GpuMat d_src1 = pool1.getBuffer(4096, 4096, CV_8UC1);   // 16MB
//    GpuMat d_dst1 = pool1.getBuffer(4096, 4096, CV_8UC3);   // 48MB, pool1 is full
//
//    GpuMat d_src2 = pool2.getBuffer(1024, 1024, CV_8UC1);   // 1MB
//    GpuMat d_dst2 = pool2.getBuffer(1024, 1024, CV_8UC3);   // 3MB
//
//    cvtColor(d_src1, d_dst1, CV_GRAY2BGR, 0, stream1);
//    cvtColor(d_src2, d_dst2, CV_GRAY2BGR, 0, stream2);
//
//    d_src1.release();
//
//    //GpuMat d_extra1 = pool1.getBuffer(1024, 1024, CV_8UC1); // 1MB, since poo1 is full, this is allocated with the DefaultAllocator
//
//    //Stream stream3;
//    //BufferPool pool3(stream3);
//
//    //GpuMat d_extra3 = pool3.getBuffer(1024, 1024, CV_8UC1);
//}

//int main()
//{
//    setBufferPoolUsage(true);                               // Tell OpenCV that we are going to utilize BufferPool
//    setBufferPoolConfig(getDevice(), 1024 * 1024 * 64, 2);  // Allocate 64 MB, 2 stacks (default is 10 MB, 5 stacks)
//
//    Stream stream1, stream2;                                // Each stream uses 1 stack
//    BufferPool pool1(stream1), pool2(stream2);
//
//    for (int i = 0; i < 10; i++)
//    {
//        GpuMat d_src1 = pool1.getBuffer(4096, 4096, CV_8UC1);   // 16MB
//        GpuMat d_dst1 = pool1.getBuffer(4096, 4096, CV_8UC3);   // 48MB, pool1 is now full
//
//        GpuMat d_src2 = pool2.getBuffer(1024, 1024, CV_8UC1);   // 1MB
//        GpuMat d_dst2 = pool2.getBuffer(1024, 1024, CV_8UC3);   // 3MB
//
//        d_src1.setTo(Scalar(i), stream1);
//        d_src2.setTo(Scalar(i), stream2);
//
//        cvtColor(d_src1, d_dst1, CV_GRAY2BGR, 0, stream1);
//        cvtColor(d_src2, d_dst2, CV_GRAY2BGR, 0, stream2);
//        // Order of destruction of the local variables is:
//        //   d_dst2 => d_src2 => d_dst1 => d_src1
//        // LIFO rule is satisfied, this code runs without error
//    }
//}

//int main()
//{
//    setBufferPoolUsage(true);                               // Tell OpenCV that we are going to utilize BufferPool
//    Stream stream;
//    BufferPool pool(stream);
//
//    GpuMat mat1 = pool.getBuffer(1024, 1024, CV_8UC1);      // Allocate mat1 (1MB)
//    GpuMat mat2 = pool.getBuffer(1024, 1024, CV_8UC1);      // Allocate mat2 (1MB)
//
//    mat1.release();                                         // erroneous usage : mat2 must be deallocated before mat1
//}