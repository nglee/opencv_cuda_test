#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

int main(int argc, const char* argv[])
{
	const cv::Mat input = cv::imread("D:/lenna.png", 0); //Load as grayscale

	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keypoints;
	detector.detect(input, keypoints);

	// Add results to image and save.
	cv::Mat output;
	cv::drawKeypoints(input, keypoints, output);
	cv::imwrite("D:/lenna_sift_2.4.13.2.jpg", output);

	return 0;
}