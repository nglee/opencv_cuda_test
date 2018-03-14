#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

int main(int argc, const char* argv[])
{
    const cv::Mat input = cv::imread("in.jpg", 0); //Load as grayscale

    cv::SiftFeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(input, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    cv::imwrite("out.jpg", output);

    return 0;
}
