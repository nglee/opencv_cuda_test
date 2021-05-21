#include "pch.h"

#include <array>
#include <numeric>

// cuda 11.1.1
#include <cuda_runtime_api.h>
#pragma comment(lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib/x64/cudart.lib")

// cudnn 8.0.5
#include "E:/TensorRT/cudnn-11.1-win64.8.0.5/include/cudnn.h"
#pragma comment(lib, "E:/TensorRT/cudnn-11.1-win64.8.0.5/lib/x64/cudnn64_8.lib")

void check_cuda(cudaError_t call, const char* file, const int line)
{
    if (cudaSuccess != call)
        GTEST_FAIL() << cudaGetErrorString(call) << ": " << file << "(" << std::to_string(line);
}
#define CHECK_CUDA(call) check_cuda((call), __FILE__, __LINE__)

void check_cudnn(cudnnStatus_t call, const char* file, const int line)
{
    if (CUDNN_STATUS_SUCCESS != call)
        GTEST_FAIL() << cudnnGetErrorString(call) << ": " << file << "(" << std::to_string(line);
}
#define CHECK_CUDNN(call) check_cudnn((call), __FILE__, __LINE__)

class BitExactnessTestSuite : public testing::TestWithParam<std::vector<std::array<int, 4>>>
{
public:
    static std::vector<cudnnConvolutionFwdAlgo_t> algos;

    static std::string get_algo_string(cudnnConvolutionFwdAlgo_t algo)
    {
        std::string ret;
        switch (algo)
        {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            ret.append("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM");
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            ret.append("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM");
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            ret.append("CUDNN_CONVOLUTION_FWD_ALGO_GEMM");
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            ret.append("CUDNN_CONVOLUTION_FWD_ALGO_DIRECT");
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            ret.append("UDNN_CONVOLUTION_FWD_ALGO_FFT");
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            ret.append("CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING");
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            ret.append("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD");
            break;
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            ret.append("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED");
            break;
        default:
            ret.append("UNKNOWN");
        }
        return ret;
    }

    static void SetUpTestCase()
    {
        std::cout << "SetUpTestCase\n";
    }

    static void TearDownTestCase()
    {
        std::cout << "TearDownTestCase\n";
    }
};

std::vector<cudnnConvolutionFwdAlgo_t> BitExactnessTestSuite::algos = {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
};

TEST_P(BitExactnessTestSuite, TestName)
{
    const cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    const cudnnDataType_t type = CUDNN_DATA_FLOAT;

    const size_t elem_size = (type == CUDNN_DATA_FLOAT ? 4 : 2);

    const std::array<int, 4> x_shape = GetParam()[0];
    const std::array<int, 4> w_shape = GetParam()[1];
    const int pad = GetParam()[2][0];
    const int stride = GetParam()[2][1];
    const int dilation = GetParam()[2][2];

    const size_t in_size = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<>());
    const size_t w_size = std::accumulate(w_shape.begin(), w_shape.end(), 1, std::multiplies<>());

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t x_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, format, type, x_shape[0], x_shape[1], x_shape[2], x_shape[3]));

    cudnnFilterDescriptor_t w_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, type, format, w_shape[0], w_shape[1], w_shape[2], w_shape[3]));

    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, type));

    CHECK_CUDNN(cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH));

    int n, c, h, w; // out shape
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &n, &c, &h, &w));
    const size_t out_size = n * c * h * w;
    const size_t out_byte = out_size * elem_size;

    std::vector<float> h_x(in_size);
    std::vector<float> h_w(w_size);
    std::vector<float> h_y1;

    for (size_t i = 0; i < in_size; ++i)
        h_x[i] = static_cast<float>(std::rand() % 256) - 127.0f;
    for (size_t i = 0; i < w_size; ++i)
        h_w[i] = static_cast<float>(std::abs(std::rand())) / RAND_MAX - 0.5f;

    cudnnTensorDescriptor_t y_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, format, type, n, c, h, w));

    std::cout << "Computed output shape(NCHW): " << std::to_string(n) << ", " << std::to_string(c) << ", " << std::to_string(h) << ", " << std::to_string(w) << "\n";

    void* d_x;
    void* d_w;
    void* d_y;
    CHECK_CUDA(cudaMalloc(&d_x, (size_t)x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3] * elem_size));
    CHECK_CUDA(cudaMalloc(&d_w, (size_t)w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3] * elem_size));
    CHECK_CUDA(cudaMalloc(&d_y, out_byte));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * elem_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), h_w.size() * elem_size, cudaMemcpyHostToDevice));

    for (auto algo : algos)
    {
        std::vector<float> h_y2;

        size_t workspace_byte;
        if (CUDNN_STATUS_NOT_SUPPORTED == cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_byte))
        {
            //std::cout << get_algo_string(algo) << ": The combination of the tensor descriptors, filter descriptor and convolution descriptor is not supported for the specified algorithm.\n";
            continue;
        }

        std::cout << "Workspace size: " << (workspace_byte / 1048576.0) << " MB, algo: " << get_algo_string(algo) << "\n";

        void* d_workspace;
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_byte));

        const float alpha = 1, beta = 0;
        int repeat = 5;

        while (repeat--)
        {
            CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, x_desc, d_x, w_desc, d_w, conv_desc, algo, d_workspace, workspace_byte, &beta, y_desc, d_y));

            // test reproducibility
            if (h_y2.empty())
            {
                //std::cout << "recording h_y2\n";
                h_y2 = std::vector<float>(out_size);
                CHECK_CUDA(cudaMemcpy(h_y2.data(), d_y, out_byte, cudaMemcpyDeviceToHost));
            }
            else
            {
                //std::cout << "comparing with h_y2\n";
                std::vector<float> y(out_size);
                CHECK_CUDA(cudaMemcpy(y.data(), d_y, out_byte, cudaMemcpyDeviceToHost));

                size_t diffcnt = 0;
                float diffsum = 0.0f;
                float diffmax = 0.0f;

                for (int i = 0; i < out_size; ++i)
                    if (h_y2[i] != y[i])
                    {
                        ++diffcnt;
                        const float diff = std::abs(h_y2[i] - y[i]);
                        diffsum += diff;
                        if (diffmax < diff)
                            diffmax = diff;
                    }

                if (diffcnt)
                {
                    const std::string msg = "# diff pixels: " + std::to_string(diffcnt) + " with average difference: " + std::to_string(diffsum / diffcnt) + " and max difference: " + std::to_string(diffmax);
                    GTEST_NONFATAL_FAILURE_(msg.c_str());
                }
            }
        }

        // test bit-exactness among different algorithms
        if (h_y1.empty())
        {
            //std::cout << "recording h_y1\n";
            h_y1 = std::vector<float>(out_size);
            CHECK_CUDA(cudaMemcpy(h_y1.data(), d_y, out_byte, cudaMemcpyDeviceToHost));
        }
        else
        {
            //std::cout << "comparing with h_y1\n";
            std::vector<float> y(out_size);
            CHECK_CUDA(cudaMemcpy(y.data(), d_y, out_byte, cudaMemcpyDeviceToHost));

            size_t diffcnt = 0;
            float diffsum = 0.0f;
            float diffmax = 0.0f;

            for (int i = 0; i < out_size; ++i)
                if (h_y1[i] != y[i])
                {
                    ++diffcnt;
                    const float diff = std::abs(h_y1[i] - y[i]);
                    diffsum += diff;
                    if (diffmax < diff)
                        diffmax = diff;
                }

            if (diffcnt)
            {
                const std::string msg = "# diff pixels: " + std::to_string(diffcnt) + " with average difference: " + std::to_string(diffsum / diffcnt) + " and max difference: " + std::to_string(diffmax);
                GTEST_NONFATAL_FAILURE_(msg.c_str());
            }
        }

        CHECK_CUDA(cudaFree(d_workspace));
    }

    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_x));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(w_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroy(handle));
}

INSTANTIATE_TEST_CASE_P(
    cudnn_conv_forward, BitExactnessTestSuite,
    testing::Values(
        std::vector<std::array<int, 4>>{{1, 3, 256, 256}, {64, 3, 7, 7}, {3, 2, 1, 0}}, // {x shape}, {w shape} {pad, stride, dilation, (not used)}
        std::vector<std::array<int, 4>>{{1, 64, 64, 64}, {64, 64, 3, 3}, {1, 1, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 64, 64, 64}, {128, 64, 3, 3}, {1, 2, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 128, 32, 32}, {128, 128, 3, 3}, {1, 1, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 128, 32, 32}, {256, 128, 3, 3}, {1, 2, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 256, 16, 16}, {256, 256, 3, 3}, {1, 1, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 256, 16, 16}, {512, 256, 3, 3}, {1, 2, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 512, 8, 8}, {512, 512, 3, 3}, {1, 1, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 64, 64, 64}, {128, 64, 1, 1}, {0, 2, 1, 0}}, // residual path
        std::vector<std::array<int, 4>>{{1, 128, 32, 32}, {256, 128, 1, 1}, {0, 2, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 256, 16, 16}, {512, 256, 1, 1}, {0, 2, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 512, 8, 8}, {11, 512, 3, 3}, {1, 1, 1, 0}}, // mapping
        std::vector<std::array<int, 4>>{{1, 256, 16, 16}, {11, 256, 3, 3}, {1, 1, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 128, 32, 32}, {11, 128, 3, 3}, {1, 1, 1, 0}},
        std::vector<std::array<int, 4>>{{1, 64, 64, 64}, {11, 64, 3, 3}, {1, 1, 1, 0}}
    ),
    [](const testing::TestParamInfo<BitExactnessTestSuite::ParamType>& info)
    {
        std::stringstream ss;
        ss << "in";
        std::for_each(info.param[0].begin(), info.param[0].end(), [&ss](int a){ ss << "_" << std::to_string(a); });
        ss << "_kernel";
        std::for_each(info.param[1].begin(), info.param[1].end(), [&ss](int a){ ss << "_" << std::to_string(a); });
        ss << "_pad_" << info.param[2][0] << "_stride_" << info.param[2][1] << "_dilation_" << info.param[2][2];
        return ss.str();
    }
);

//class BitExactnessTestSuite : public testing::TestWithParam<cudnnConvolutionFwdAlgo_t>
//{
//public:
//  static std::vector<int> x_shape;
//  static std::vector<int> w_shape;
//
//  static std::vector<float> h_x;
//  static std::vector<float> h_w;
//  static std::vector<float> h_y;
//
//  static void SetUpTestCase()
//  {
//      //x_shape = {1, 3, 256, 256};
//      //w_shape = {64, 3, 7, 7};
//      x_shape = {1, 64, 64, 64};
//      w_shape = {64, 64, 3, 3};
//
//      const size_t in_size = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<>());
//      const size_t w_size = std::accumulate(w_shape.begin(), w_shape.end(), 1, std::multiplies<>());
//
//      h_x = std::vector<float>(in_size);
//      h_w = std::vector<float>(w_size);
//
//      for (size_t i = 0; i < in_size; ++i)
//          h_x[i] = static_cast<float>(std::rand() % 256) - 127.0f;
//      for (size_t i = 0; i < w_size; ++i)
//          h_w[i] = static_cast<float>(std::abs(std::rand())) / RAND_MAX - 0.5f;
//  }
//
//  static void TearDownTestCase()
//  {
//  }
//};
//
//std::vector<int> BitExactnessTestSuite::x_shape;
//std::vector<int> BitExactnessTestSuite::w_shape;
//
//std::vector<float> BitExactnessTestSuite::h_x;
//std::vector<float> BitExactnessTestSuite::h_w;
//std::vector<float> BitExactnessTestSuite::h_y;
//
//TEST_P(BitExactnessTestSuite, TestName)
//{
//  const cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
//  const cudnnDataType_t type = CUDNN_DATA_FLOAT;
//  const cudnnConvolutionFwdAlgo_t algo = GetParam();
//
//  const size_t elem_size = (type == CUDNN_DATA_FLOAT ? 4 : 2);
//
//  cudnnHandle_t handle;
//  CHECK_CUDNN(cudnnCreate(&handle));
//
//  cudnnTensorDescriptor_t x_desc;
//  CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
//  CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, format, type, x_shape[0], x_shape[1], x_shape[2], x_shape[3]));
//
//  cudnnFilterDescriptor_t w_desc;
//  CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
//  CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, type, format, w_shape[0], w_shape[1], w_shape[2], w_shape[3]));
//
//  cudnnConvolutionDescriptor_t conv_desc;
//  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
//  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, type));
//
//  int n, c, h, w; // out shape
//  CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &n, &c, &h, &w));
//  const size_t out_elems = n * c * h * w;
//  const size_t out_byte = out_elems * elem_size;
//
//  cudnnTensorDescriptor_t y_desc;
//  CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
//  CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, format, type, n, c, h, w));
//
//  size_t workspace_byte;
//  auto status = cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_byte);
//  if (CUDNN_STATUS_NOT_SUPPORTED == status)
//  {
//      //std::cout << "The combination of the tensor descriptors, filter descriptor and convolution descriptor is not supported for the specified algorithm: " << std::to_string(algo) << "\n";
//      goto END1;
//  }
//
//  std::cout << "Computed output shape(NCHW): " << std::to_string(n) << ", " << std::to_string(c) << ", " << std::to_string(h) << ", " << std::to_string(w) << "\n";
//  std::cout << "Workspace size: " << (workspace_byte / 1048576.0) << " MB\n";
//
//  void* d_workspace;
//  void* d_x;
//  void* d_w;
//  void* d_y;
//  CHECK_CUDA(cudaMalloc(&d_workspace, workspace_byte));
//  CHECK_CUDA(cudaMalloc(&d_x, (size_t)x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3] * elem_size));
//  CHECK_CUDA(cudaMalloc(&d_w, (size_t)w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3] * elem_size));
//  CHECK_CUDA(cudaMalloc(&d_y, out_byte));
//
//  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));
//  CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice));
//
//  const float alpha = 1, beta = 0;
//  CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, x_desc, d_x, w_desc, d_w, conv_desc, algo, d_workspace, workspace_byte, &beta, y_desc, d_y));
//
//  if (h_y.empty())
//  {
//      std::cout << "recording h_y\n";
//      h_y = std::vector<float>(out_elems);
//      CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, out_byte, cudaMemcpyDeviceToHost));
//  }
//  else
//  {
//      std::cout << "comparing with h_y\n";
//      std::vector<float> y(out_elems);
//      CHECK_CUDA(cudaMemcpy(y.data(), d_y, out_byte, cudaMemcpyDeviceToHost));
//
//      size_t diffcnt = 0;
//      float diffsum = 0.0f;
//      float diffmax = 0.0f;
//
//      for (int i = 0; i < out_elems; ++i)
//          if (h_y[i] != y[i])
//          {
//              ++diffcnt;
//              const float diff = std::abs(h_y[i] - y[i]);
//              diffsum += diff;
//              if (diffmax < diff)
//                  diffmax = diff;
//          }
//
//      if (diffcnt)
//          GTEST_FAIL() << "# diff pixels: " << std::to_string(diffcnt) << " with average difference: " << diffsum / diffcnt << " and max difference: " << diffmax;
//  }
//
//  CHECK_CUDA(cudaFree(d_y));
//  CHECK_CUDA(cudaFree(d_w));
//  CHECK_CUDA(cudaFree(d_x));
//  CHECK_CUDA(cudaFree(d_workspace));
//END1:
//  CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
//  CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
//  CHECK_CUDNN(cudnnDestroyFilterDescriptor(w_desc));
//  CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
//  CHECK_CUDNN(cudnnDestroy(handle));
//}
//
//INSTANTIATE_TEST_CASE_P(
//  cudnn_conv_forward, BitExactnessTestSuite,
//  testing::Values(
//      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
//      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
//      CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
//      CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
//      CUDNN_CONVOLUTION_FWD_ALGO_FFT,
//      CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
//      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
//      CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
//  ),
//  [](const testing::TestParamInfo<BitExactnessTestSuite::ParamType>& info)
//  {
//      std::string ret;
//      switch(info.param)
//      {
//      case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
//          ret.append("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM");
//          break;
//      case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
//          ret.append("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM");
//          break;
//      case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
//          ret.append("CUDNN_CONVOLUTION_FWD_ALGO_GEMM");
//          break;
//      case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
//          ret.append("CUDNN_CONVOLUTION_FWD_ALGO_DIRECT");
//          break;
//      case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
//          ret.append("UDNN_CONVOLUTION_FWD_ALGO_FFT");
//          break;
//      case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
//          ret.append("CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING");
//          break;
//      case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
//          ret.append("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD");
//          break;
//      case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
//          ret.append("CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED");
//          break;
//      default:
//          ret.append("UNKNOWN");
//      }
//      return ret;
//  }
//);