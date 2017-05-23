#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

int main(int argc, const char* argv[])
{
	const cv::Mat input = cv::imread("D:/lenna.png", 0); //Load as grayscale

	cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SiftFeatureDetector::create();
	std::vector<cv::KeyPoint> keypoints;
	f2d->detect(input, keypoints);

	cv::Mat output;
	cv::drawKeypoints(input, keypoints, output);
	cv::imwrite("D:/lenna_sift_3.2.jpg", output);

	return 0;
}