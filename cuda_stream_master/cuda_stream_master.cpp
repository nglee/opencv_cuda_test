#include <opencv2/opencv.hpp>

#define TEST_NUM        2               // Change test number

#define CONCAT(a, b)    a ## b
#define CALL_TEST(n)    CONCAT(test_, n) ## ()

using namespace cv;
using namespace cv::cuda;

void test_0()
{
    setBufferPoolUsage(true);

    {
        BufferPool pool(Stream::Null());
        GpuMat d = pool.getBuffer(512, 512, CV_8UC1);
        d.setTo(Scalar(0));
    }

    resetDevice();

    {
        BufferPool pool(Stream::Null());
        GpuMat d = pool.getBuffer(512, 512, CV_8UC1);
        d.setTo(Scalar(255));
    }
}

void test_1()
{
    setBufferPoolUsage(true);

    {
        BufferPool pool(Stream::Null());
        GpuMat d = pool.getBuffer(512, 512, CV_8UC1);
        d.setTo(Scalar(0));
    }

    resetDevice();

    {
        Stream stream;
        BufferPool pool(stream);
        GpuMat d = pool.getBuffer(512, 512, CV_8UC1);
        d.setTo(Scalar(255));
    }
}

void test_2()
{
    setBufferPoolUsage(true);

    BufferPool pool(Stream::Null());

    Mat test1024x512 = imread("./test512x1024.png", IMREAD_GRAYSCALE); // This is an 512(col)x1024(row) image
    Mat test512x512T = imread("./test512x512T.png", IMREAD_GRAYSCALE); // This is the top half of the above image, so the size is 512(col)x512(row)

    GpuMat d1 = pool.getBuffer(512, 512, CV_8UC1); // (1)
    GpuMat d2 = pool.getBuffer(512, 512, CV_8UC1); // (2)
    d2.upload(test512x512T);

    d1.release();           // ***** Here we violate the deallocation rule when the StackAllocator is enabled. *****
                            // ***** The memory allocated on line (1) is deallocated before the memory allocated on line (2). *****
                            // ***** In debug mode, the program stops on this line. In release mode, it doesn't, but let's see what happens. *****

    Mat out;
    d2.download(out);
    imshow("out1", out);    // We expect to see test512x512T.png, and indeed, the expected image will be displayed.

    GpuMat d3 = pool.getBuffer(1024, 512, CV_8UC1);
    d3.upload(test1024x512);

    d2.download(out);
    imshow("out2", out);    // We expect to see test512x512T.png, but the bottom half of test512x1024.png will be displayed.
    waitKey(0);
}

#include <omp.h>

void test_3()
{
    setBufferPoolUsage(true);

    #pragma omp parallel num_threads(64)
    {
        Mat h;
        h = imread("./test512x1024.png", IMREAD_GRAYSCALE);

        GpuMat d_(h.size(), CV_8UC1);
        GpuMat d(h.size(), CV_64FC1);

        d_.upload(h);
        d_.convertTo(d, CV_64F, 1.0 / 255.0);

        GpuMat out;
        cv::cuda::calcNorm(d, out, NORM_L2);

        Mat h_out;
        out.download(h_out);

        #pragma omp critical
        {
            std::cout << "(" << std::setw(2) << omp_get_thread_num() << "): " << h_out.at<double>(0, 0) << std::endl;
        }
    }
}

int main()
{
    CALL_TEST(TEST_NUM);
}
