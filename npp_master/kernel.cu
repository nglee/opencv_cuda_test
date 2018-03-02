#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "npp.h"
#include "nppi.h"

#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <stdio.h>

using namespace cv;
using namespace cv::cuda;

    #define error_entry(entry)  { entry, #entry }

    struct ErrorEntry
    {
        int code;
        const char* str;
    };

    struct ErrorEntryComparer
    {
        int code;
        ErrorEntryComparer(int code_) : code(code_) {}
        bool operator()(const ErrorEntry& e) const { return e.code == code; }
    };

    const ErrorEntry npp_errors [] =
    {
    #if defined (_MSC_VER)
        error_entry( NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY ),
    #endif


        error_entry( NPP_INVALID_HOST_POINTER_ERROR ),
        error_entry( NPP_INVALID_DEVICE_POINTER_ERROR ),
        error_entry( NPP_LUT_PALETTE_BITSIZE_ERROR ),
        error_entry( NPP_ZC_MODE_NOT_SUPPORTED_ERROR ),
        error_entry( NPP_MEMFREE_ERROR ),
        error_entry( NPP_MEMSET_ERROR ),
        error_entry( NPP_QUALITY_INDEX_ERROR ),
        error_entry( NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR ),
        error_entry( NPP_CHANNEL_ORDER_ERROR ),
        error_entry( NPP_ZERO_MASK_VALUE_ERROR ),
        error_entry( NPP_QUADRANGLE_ERROR ),
        error_entry( NPP_RECTANGLE_ERROR ),
        error_entry( NPP_COEFFICIENT_ERROR ),
        error_entry( NPP_NUMBER_OF_CHANNELS_ERROR ),
        error_entry( NPP_COI_ERROR ),
        error_entry( NPP_DIVISOR_ERROR ),
        error_entry( NPP_CHANNEL_ERROR ),
        error_entry( NPP_STRIDE_ERROR ),
        error_entry( NPP_ANCHOR_ERROR ),
        error_entry( NPP_MASK_SIZE_ERROR ),
        error_entry( NPP_MIRROR_FLIP_ERROR ),
        error_entry( NPP_MOMENT_00_ZERO_ERROR ),
        error_entry( NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR ),
        error_entry( NPP_THRESHOLD_ERROR ),
        error_entry( NPP_CONTEXT_MATCH_ERROR ),
        error_entry( NPP_FFT_FLAG_ERROR ),
        error_entry( NPP_FFT_ORDER_ERROR ),
        error_entry( NPP_SCALE_RANGE_ERROR ),
        error_entry( NPP_DATA_TYPE_ERROR ),
        error_entry( NPP_OUT_OFF_RANGE_ERROR ),
        error_entry( NPP_DIVIDE_BY_ZERO_ERROR ),
        error_entry( NPP_MEMORY_ALLOCATION_ERR ),
        error_entry( NPP_RANGE_ERROR ),
        error_entry( NPP_BAD_ARGUMENT_ERROR ),
        error_entry( NPP_NO_MEMORY_ERROR ),
        error_entry( NPP_ERROR_RESERVED ),
        error_entry( NPP_NO_OPERATION_WARNING ),
        error_entry( NPP_DIVIDE_BY_ZERO_WARNING ),
        error_entry( NPP_WRONG_INTERSECTION_ROI_WARNING ),

        error_entry( NPP_NOT_SUPPORTED_MODE_ERROR ),
        error_entry( NPP_ROUND_MODE_NOT_SUPPORTED_ERROR ),
        error_entry( NPP_RESIZE_NO_OPERATION_ERROR ),
        error_entry( NPP_LUT_NUMBER_OF_LEVELS_ERROR ),
        error_entry( NPP_TEXTURE_BIND_ERROR ),
        error_entry( NPP_WRONG_INTERSECTION_ROI_ERROR ),
        error_entry( NPP_NOT_EVEN_STEP_ERROR ),
        error_entry( NPP_INTERPOLATION_ERROR ),
        error_entry( NPP_RESIZE_FACTOR_ERROR ),
        error_entry( NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR ),
        error_entry( NPP_MEMCPY_ERROR ),
        error_entry( NPP_ALIGNMENT_ERROR ),
        error_entry( NPP_STEP_ERROR ),
        error_entry( NPP_SIZE_ERROR ),
        error_entry( NPP_NULL_POINTER_ERROR ),
        error_entry( NPP_CUDA_KERNEL_EXECUTION_ERROR ),
        error_entry( NPP_NOT_IMPLEMENTED_ERROR ),
        error_entry( NPP_ERROR ),
        error_entry( NPP_NO_ERROR ),
        error_entry( NPP_SUCCESS ),
        error_entry( NPP_WRONG_INTERSECTION_QUAD_WARNING ),
        error_entry( NPP_MISALIGNED_DST_ROI_WARNING ),
        error_entry( NPP_AFFINE_QUAD_INCORRECT_WARNING ),
        error_entry( NPP_DOUBLE_SIZE_WARNING )
    };

    const size_t npp_error_num = sizeof(npp_errors) / sizeof(npp_errors[0]);

    cv::String getErrorString(int code, const ErrorEntry* errors, size_t n)
    {
        size_t idx = std::find_if(errors, errors + n, ErrorEntryComparer(code)) - errors;

        const char* msg = (idx != n) ? errors[idx].str : "Unknown error code";
        cv::String str = cv::format("%s [Code = %d]", msg, code);

        return str;
    }

String getNppErrorMessage(int code)
{
    return getErrorString(code, npp_errors, npp_error_num);
}

static inline void checkNppError(int code, const char* file, const int line, const char* func)
{
    if (code < 0)
        cv::error(cv::Error::GpuApiCallError, getNppErrorMessage(code), func, file, line);
}

#define nppSafeCall(expr)  checkNppError(expr, __FILE__, __LINE__, CV_Func)

int main()
{
    #pragma omp parallel num_threads(2)
    {
        // Source image : 512x512 1channel 8bit, all pixels set to 5
        GpuMat src(512, 512, CV_8UC1);
        src.setTo(Scalar(5));

        // Destination buffer where mean and stddev value is stored
        GpuMat dst(1, 2, CV_64FC1);

        // Create scratch buffer
        int bufSize;
        NppiSize sz;
        sz.width = src.cols;
        sz.height = src.rows;
        nppSafeCall( nppiMeanStdDevGetBufferHostSize_8u_C1R(sz, &bufSize) );
        GpuMat buf(1, bufSize, CV_8UC1);
        
        // Create stream
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Set npp to use this stream
        nppSetStream(stream);

        nppSafeCall( nppiMean_StdDev_8u_C1R(src.ptr<Npp8u>(), static_cast<int>(src.step), sz, buf.ptr<Npp8u>(), dst.ptr<Npp64f>(), dst.ptr<Npp64f>() + 1) );

        // Wait until npp finish
        cudaStreamSynchronize(stream);

        // Destroy stream
        cudaStreamDestroy(stream);

        // Print output (expects mean = 5, stddev = 0)
        Mat h_dst;
        dst.download(h_dst);
        std::string out = "thread" + std::to_string(omp_get_thread_num()) + " : (mean)" + std::to_string(h_dst.at<Npp64f>(0, 0)) + " (stddev)" + std::to_string(h_dst.at<Npp64f>(0, 1)) + "\n";
        std::cout << out;
    }
}