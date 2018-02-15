#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;
using namespace cv::cuda;

//#if !_DEBUG
//// run in release mode : second setTo => invalid memory access
//int main()
//{
//	setBufferPoolUsage(true);
//
//	BufferPool pool(Stream::Null());
//
//	GpuMat d = pool.getBuffer(512, 512, CV_8UC1);
//	d.setTo(Scalar(0));
//
//	resetDevice();
//
//	BufferPool pool2(Stream::Null());
//
//	d = pool2.getBuffer(512, 512, CV_8UC1);
//	d.setTo(Scalar(255));
//}
//#else
//
//// run in debug mode : second copy assignment => CV_ASSERT
//int main()
//{
//	setBufferPoolUsage(true);
//
//	BufferPool pool(Stream::Null());
//
//	GpuMat d = pool.getBuffer(512, 512, CV_8UC1);
//	GpuMat d2 = d;
//	d = pool.getBuffer(512, 512, CV_8UC1);
//}
//#endif

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

//int main()
//{
//    setBufferPoolUsage(false);
//
//    Stream stream;
//
//    BufferPool pool(stream);
//
//    GpuMat d_src = pool.getBuffer(256, 256, CV_8UC1);
//    GpuMat d_dst = pool.getBuffer(256, 256, CV_8UC3);
//
//    cv::cuda::cvtColor(d_src, d_dst, CV_GRAY2BGR, 0, stream);
//
//    setBufferPoolUsage(true);
//
//    Stream stream2;
//    BufferPool pool2(stream2);
//    GpuMat d_src2 = pool2.getBuffer(256, 256, CV_8UC1);
//    GpuMat d_dst2 = pool2.getBuffer(256, 256, CV_8UC3);
//
//    cv::cuda::cvtColor(d_src2, d_dst2, CV_GRAY2BGR, 0, stream);
//}