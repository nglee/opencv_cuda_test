#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::cuda;

int main()
{
    setBufferPoolUsage(true);
    Stream stream;
    BufferPool pool(stream);

    {
        GpuMat mat = pool.getBuffer(424, 512, CV_16UC1);
        mat.setTo(Scalar(5), stream);
    }
}