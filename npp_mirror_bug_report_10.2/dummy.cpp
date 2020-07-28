#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <nppi.h>
#include <opencv2/opencv.hpp>

#include <cstdio>

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

void test_8UC3(size_t, size_t);
void test_8UC4(size_t, size_t);
void test_16UC3(size_t, size_t);
void test_16UC4(size_t, size_t);
void test_32FC3(size_t h, size_t w);
void test_32FC4(size_t h, size_t w);

int main()
{
    for (int i = 128; i < 2048; i += 29)
        for (int j = 125; j < 2048; j += 31)
        {
            test_8UC3(i, j);
            test_8UC4(i, j);
            test_16UC3(i, j);
            test_16UC4(i, j);
            test_32FC3(i, j);
            test_32FC4(i, j);
        }
}

void test_8UC3(const size_t h, const size_t w)
{
    printf("test_8UC3 case h: %llu, w: %llu\n", h, w);

    cv::Mat h_src(h, w, CV_8UC3, cv::Scalar(1, 1, 1));

    // Fill src data
    {
        cv::Mat h_src2, h_src3, h_src4;
        cv::integral(h_src, h_src2);
        h_src2.convertTo(h_src3, CV_8UC3, 255 / (float)(w * h));
        h_src4 = h_src3;
        h_src = h_src4(cv::Rect(1, 1, w, h)).clone();
    }

    // Copy host data to device
    cv::cuda::GpuMat d_srcDst(h, w, CV_8UC3);
    d_srcDst.upload(h_src);

    NppiSize roi;
    roi.width = w;
    roi.height = h;

    // Call npp API
    NppStatus status = nppiMirror_8u_C3IR((Npp8u*)d_srcDst.data, d_srcDst.step, roi, NPP_BOTH_AXIS);
    if (status != NPP_NO_ERROR)
        printf("npp error\n");

    // Copy result device data to host
    cv::Mat h_dst;
    d_srcDst.download(h_dst);

    // Check result
    for (size_t _w = 0; _w < w; _w++)
        for (size_t _h = 0; _h < h; _h++)
        {
            uchar3 src = h_src.ptr<uchar3>(h - 1 -_h)[w - 1 - _w]; // both-axis flip
            uchar3 dst = h_dst.ptr<uchar3>(_h)[_w];
            if (src.x != dst.x
                || src.y != dst.y
                || src.z != dst.z)
                //|| src.w != dst.w)
            {
                printf("wrong\n");
                exit(1);
            }
        }
}

void test_8UC4(const size_t h, const size_t w)
{
    printf("test_8UC4 case h: %llu, w: %llu\n", h, w);

    cv::Mat h_src(h, w, CV_8UC4, cv::Scalar(1, 1, 1, 1));

    // Fill src data
    {
        cv::Mat h_src2, h_src3, h_src4;
        cv::integral(h_src, h_src2);
        h_src2.convertTo(h_src3, CV_8UC4, 255 / (float)(w * h));
        h_src4 = h_src3;
        h_src = h_src4(cv::Rect(1, 1, w, h)).clone();
    }

    // Copy host data to device
    cv::cuda::GpuMat d_srcDst(h, w, CV_8UC4);
    d_srcDst.upload(h_src);

    NppiSize roi;
    roi.width = w;
    roi.height = h;

    // Call npp API
    NppStatus status = nppiMirror_8u_C4IR((Npp8u*)d_srcDst.data, d_srcDst.step, roi, NPP_BOTH_AXIS);
    if (status != NPP_NO_ERROR)
        printf("npp error\n");

    // Copy result device data to host
    cv::Mat h_dst;
    d_srcDst.download(h_dst);

    // Check result
    for (size_t _w = 0; _w < w; _w++)
        for (size_t _h = 0; _h < h; _h++)
        {
            uchar4 src = h_src.ptr<uchar4>(h - 1 -_h)[w - 1 - _w]; // both-axis flip
            uchar4 dst = h_dst.ptr<uchar4>(_h)[_w];
            if (src.x != dst.x
                || src.y != dst.y
                || src.z != dst.z
                || src.w != dst.w)
            {
                printf("wrong\n");
                exit(1);
            }
        }
}

void test_16UC3(const size_t h, const size_t w)
{
    printf("test_16UC3 case h: %llu, w: %llu\n", h, w);

    cv::Mat h_src(h, w, CV_8UC3, cv::Scalar(1, 1, 1));

    // Fill src data
    {
        cv::Mat h_src2, h_src3, h_src4;
        cv::integral(h_src, h_src2);
        h_src2.convertTo(h_src3, CV_16UC3, 255 / (float)(w * h));
        h_src4 = h_src3;
        h_src = h_src4(cv::Rect(1, 1, w, h)).clone();
    }

    // Copy host data to device
    cv::cuda::GpuMat d_srcDst(h, w, CV_16UC3);
    d_srcDst.upload(h_src);

    NppiSize roi;
    roi.width = w;
    roi.height = h;

    // Call npp API
    NppStatus status = nppiMirror_16u_C3IR((Npp16u*)d_srcDst.data, d_srcDst.step, roi, NPP_BOTH_AXIS);
    if (status != NPP_NO_ERROR)
        printf("npp error\n");

    // Copy result device data to host
    cv::Mat h_dst;
    d_srcDst.download(h_dst);

    // Check result
    for (size_t _w = 0; _w < w; _w++)
        for (size_t _h = 0; _h < h; _h++)
        {
            ushort3 src = h_src.ptr<ushort3>(h - 1 -_h)[w - 1 - _w]; // both-axis flip
            ushort3 dst = h_dst.ptr<ushort3>(_h)[_w];
            if (src.x != dst.x
                || src.y != dst.y
                || src.z != dst.z)
                //|| src.w != dst.w)
            {
                printf("wrong\n");
                exit(1);
            }
        }
}

void test_16UC4(const size_t h, const size_t w)
{
    printf("test_16UC4 case h: %llu, w: %llu\n", h, w);

    cv::Mat h_src(h, w, CV_8UC4, cv::Scalar(1, 1, 1, 1));

    // Fill src data
    {
        cv::Mat h_src2, h_src3, h_src4;
        cv::integral(h_src, h_src2);
        h_src2.convertTo(h_src3, CV_16UC4, 255 / (float)(w * h));
        h_src4 = h_src3;
        h_src = h_src4(cv::Rect(1, 1, w, h)).clone();
    }

    // Copy host data to device
    cv::cuda::GpuMat d_srcDst(h, w, CV_16UC4);
    d_srcDst.upload(h_src);

    NppiSize roi;
    roi.width = w;
    roi.height = h;

    // Call npp API
    NppStatus status = nppiMirror_16u_C4IR((Npp16u*)d_srcDst.data, d_srcDst.step, roi, NPP_BOTH_AXIS);
    if (status != NPP_NO_ERROR)
        printf("npp error\n");

    // Copy result device data to host
    cv::Mat h_dst;
    d_srcDst.download(h_dst);

    // Check result
    for (size_t _w = 0; _w < w; _w++)
        for (size_t _h = 0; _h < h; _h++)
        {
            ushort4 src = h_src.ptr<ushort4>(h - 1 -_h)[w - 1 - _w]; // both-axis flip
            ushort4 dst = h_dst.ptr<ushort4>(_h)[_w];
            if (src.x != dst.x
                || src.y != dst.y
                || src.z != dst.z
                || src.w != dst.w)
            {
                printf("wrong\n");
                exit(1);
            }
        }
}

void test_32FC3(const size_t h, const size_t w)
{
    printf("test_32FC3 case h: %llu, w: %llu\n", h, w);

    cv::Mat h_src(h, w, CV_32FC3, cv::Scalar(1, 1.1, 0.9));

    // Fill src data
    {
        cv::Mat h_src2, h_src3, h_src4;
        cv::integral(h_src, h_src2);
        h_src2.convertTo(h_src3, CV_32FC3);
        h_src4 = h_src3 / (float)(w * h);
        h_src = h_src4(cv::Rect(1, 1, w, h)).clone();
    }

    // Copy host data to device
    cv::cuda::GpuMat d_srcDst(h, w, CV_32FC3);
    d_srcDst.upload(h_src);

    NppiSize roi;
    roi.width = w;
    roi.height = h;

    // Call npp API
    NppStatus status = nppiMirror_32f_C3IR((Npp32f*)d_srcDst.data, d_srcDst.step, roi, NPP_BOTH_AXIS);
    if (status != NPP_NO_ERROR)
        printf("npp error\n");

    // Copy result device data to host
    cv::Mat h_dst;
    d_srcDst.download(h_dst);

    // Check result
    for (size_t _w = 0; _w < w; _w++)
        for (size_t _h = 0; _h < h; _h++)
        {
            float3 src = h_src.ptr<float3>(h - 1 -_h)[w - 1 - _w]; // both-axis flip
            float3 dst = h_dst.ptr<float3>(_h)[_w];
            if (src.x != dst.x
                || src.y != dst.y
                || src.z != dst.z)
                //|| src.w != dst.w)
            {
                printf("wrong\n");
                exit(1);
            }
        }
}

void test_32FC4(const size_t h, const size_t w)
{
    printf("test_32FC4 case h: %llu, w: %llu\n", h, w);

    cv::Mat h_src(h, w, CV_32FC4, cv::Scalar(1, 1.1, 1, 0.9));

    // Fill src data
    {
        cv::Mat h_src2, h_src3, h_src4;
        cv::integral(h_src, h_src2);
        h_src2.convertTo(h_src3, CV_32FC4);
        h_src4 = h_src3 / (float)(w * h);
        h_src = h_src4(cv::Rect(1, 1, w, h)).clone();
    }

    // Copy host data to device
    cv::cuda::GpuMat d_srcDst(h, w, CV_32FC4);
    d_srcDst.upload(h_src);

    NppiSize roi;
    roi.width = w;
    roi.height = h;

    // Call npp API
    NppStatus status = nppiMirror_32f_C4IR((Npp32f*)d_srcDst.data, d_srcDst.step, roi, NPP_BOTH_AXIS);
    if (status != NPP_NO_ERROR)
        printf("npp error\n");

    // Copy result device data to host
    cv::Mat h_dst;
    d_srcDst.download(h_dst);

    // Check result
    for (size_t _w = 0; _w < w; _w++)
        for (size_t _h = 0; _h < h; _h++)
        {
            float4 src = h_src.ptr<float4>(h - 1 -_h)[w - 1 - _w]; // both-axis flip
            float4 dst = h_dst.ptr<float4>(_h)[_w];
            if (src.x != dst.x
                || src.y != dst.y
                || src.z != dst.z
                || src.w != dst.w)
            {
                printf("wrong\n");
                exit(1);
            }
        }
}
