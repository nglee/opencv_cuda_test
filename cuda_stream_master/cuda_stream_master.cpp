#include <opencv2/opencv.hpp>

#define TEST_NUM        3               // Change test number

#define CONCAT(a, b)    a ## b
#define CALL_TEST(n)    CONCAT(test_, n) ## ()

using namespace cv;
using namespace cv::cuda;

void test_0()
{
    BufferPool pool(Stream::Null());
    GpuMat d = pool.getBuffer(512, 512, CV_8UC1);
    d.setTo(Scalar(0));

    resetDevice();

    BufferPool pool2(Stream::Null());
    d = pool2.getBuffer(512, 512, CV_8UC1);
    d.setTo(Scalar(255));
}

void test_1()
{
    Stream stream = Stream::Null();
    BufferPool pool(stream);
    GpuMat d = pool.getBuffer(512, 512, CV_8UC1);
    d.setTo(Scalar(0));

    resetDevice();

    BufferPool pool2(stream);
    d = pool2.getBuffer(512, 512, CV_8UC1);
    d.setTo(Scalar(255));
}

void test_2()
{
    {
        BufferPool pool(Stream::Null());
        GpuMat d = pool.getBuffer(512, 512, CV_8UC1);
        d.setTo(Scalar(0));
    }

    resetDevice();

    {
        BufferPool pool2(Stream::Null());
        GpuMat d = pool2.getBuffer(512, 512, CV_8UC1);
        d.setTo(Scalar(255));
    }
}

void test_3()
{
    setBufferPoolUsage(true);

    BufferPool pool(Stream::Null());

    Mat test1024x512 = imread("./test512x1024.png", IMREAD_GRAYSCALE); // This is an 512(col)x1024(row) image
    Mat test512x512T = imread("./test512x512T.png", IMREAD_GRAYSCALE); // This is the top half of the above image, so the size is 512(col)x512(row)

    GpuMat d = pool.getBuffer(512, 512, CV_8UC1); // (1)
    GpuMat d2 = d;
    d = pool.getBuffer(512, 512, CV_8UC1);        // (2)
    d.upload(test512x512T);

    d2.release();          // ***** Here we violate the deallocation rule when the StackAllocator is enabled. *****
                           // ***** That is, the memory allocated on line (1) is deallocated before the memory allocated on line (2). *****
                           // ***** In debug mode, the program stops on this line. In release mode, it doesn't, but let's see what happens. *****

    Mat out;
    d.download(out);
    imshow("out", out);    // We expect to see test512x512T.png, and indeed, the expected image will be displayed.
    waitKey(0);

    GpuMat d3 = pool.getBuffer(1024, 512, CV_8UC1);
    d3.upload(test1024x512);

    d.download(out);
    imshow("out", out);    // We expect to see test512x512T.png, but contrary to our expectation, the bottom half of test512x1024.png will be displayed.
    waitKey(0);
}

int main()
{
    setBufferPoolUsage(true);
    CALL_TEST(TEST_NUM);
}