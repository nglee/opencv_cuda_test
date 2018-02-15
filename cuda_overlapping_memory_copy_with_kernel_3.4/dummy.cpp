#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace cv::cuda;

int main()
{
    Mat src_;
    src_ = imread("E://Cropper_Image_01.png", IMREAD_GRAYSCALE);

    HostMem src(src_);

    Ptr<cv::cuda::FastFeatureDetector> d_FAST[2];
    d_FAST[0] = cv::cuda::FastFeatureDetector::create(20);
    d_FAST[1] = cv::cuda::FastFeatureDetector::create(20);

    GpuMat d_src[2];
    d_src[0] = GpuMat(src.rows,src.cols,CV_8UC4);
    d_src[1] = GpuMat(src.rows,src.cols,CV_8UC4);

    GpuMat d_keypoints[2];

    #pragma omp parallel num_threads(2)
    {
        Stream stream;
        int threadNum = omp_get_thread_num();
        d_src[threadNum].upload(src, stream);
        d_FAST[threadNum]->detectAsync(d_src[threadNum], d_keypoints[threadNum], noArray(), stream);
    }
}