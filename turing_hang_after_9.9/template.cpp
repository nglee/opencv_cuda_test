#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

int main()
{
    cv::cuda::GpuMat image_h(1083, 1923, CV_8UC1);
    cv::cuda::GpuMat templ_h(77, 70, CV_8UC1);
    cv::cuda::GpuMat result(1083-77+1, 1923-70+1, CV_32FC1);

    printf("1\n");
    cv::Ptr<cv::cuda::TemplateMatching> alg;
    alg = cv::cuda::createTemplateMatching(templ_h.type(), CV_TM_CCOEFF_NORMED);
    printf("2\n");
    alg->match(image_h, templ_h, result);
    printf("3\n");

    cudaDeviceSynchronize();
    printf("4\n");
}
