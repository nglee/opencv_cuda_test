#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[])
{
    const std::string fname = "rtsp://admin:admin@192.168.1.13/media/video2";

    cv::namedWindow("GPU", cv::WINDOW_NORMAL);

    cv::cuda::GpuMat d_frame;
    cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);

    for (;;)
    {

        if (!d_reader->nextFrame(d_frame))
            break;

        cv::Mat frame;
        d_frame.download(frame);
        cv::imshow("GPU", frame);

        if (cv::waitKey(3) > 0)
            break;
    }
    return 0;
}