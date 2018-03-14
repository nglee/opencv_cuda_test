#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

int main(int argc, const char* argv[])
{
    const cv::Mat input = cv::imread("in.jpg", 0); //Load as grayscale

    cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    f2d->detect(input, keypoints);

    // Add results to image and save
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    cv::imwrite("out.jpg", output);

    return 0;
}
