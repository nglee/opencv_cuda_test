#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat mat(800, 1200, CV_8UC3, cv::Scalar(128, 128, 128));

    cv::Point p1(200, 100);
    cv::Point p2(1000, 700);

    const cv::Scalar red(0, 0, 255);

    cv::LineIterator it(mat, p1, p2);
    for (int i = 0; i < it.count; i++, it++)
        if (i % 30 < 20) {
            cv::Vec3b* p = (cv::Vec3b *)(*it);
            (*p)[0] = red[0];
            (*p)[1] = red[1];
            (*p)[2] = red[2];
        }

    cv::imshow("test", mat);
    cv::waitKey();
}