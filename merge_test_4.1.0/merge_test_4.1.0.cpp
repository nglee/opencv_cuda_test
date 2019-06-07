#include <opencv2/opencv.hpp>
#include <vector>

void one_channel_test()
{
	cv::Mat a(2, 2, CV_8UC1);
	cv::Mat b(2, 2, CV_8UC1);
	cv::Mat c(2, 2, CV_8UC1);

	a.at<uchar>(0, 0) = 0; a.at<uchar>(0, 1) = 1;
	a.at<uchar>(1, 0) = 2; a.at<uchar>(1, 1) = 3;
	b.at<uchar>(0, 0) = 4; b.at<uchar>(0, 1) = 5;
	b.at<uchar>(1, 0) = 6; b.at<uchar>(1, 1) = 7;
	c.at<uchar>(0, 0) = 8; c.at<uchar>(0, 1) = 9;
	c.at<uchar>(1, 0) = 10; c.at<uchar>(1, 1) = 11;

	cv::Mat merged;
	cv::merge(std::vector<cv::Mat>({a, b, c }), merged);

	for (int y = 0; y < merged.rows; ++y)
	{
		for (int x = 0; x < merged.cols; ++x)
		{
			std::cout << "{ ";
			std::cout << std::to_string(merged.at<uchar[3]>(y, x)[0]) << " ";
			std::cout << std::to_string(merged.at<uchar[3]>(y, x)[1]) << " ";
			std::cout << std::to_string(merged.at<uchar[3]>(y, x)[2]) << "} ";
		}
		std::cout << "\n";
	}
}

void multi_channel_test()
{
	cv::Mat a(2, 2, CV_8UC3);
	cv::Mat b(2, 2, CV_8UC3);
	cv::Mat c(2, 2, CV_8UC3);

	(a.at<uchar[3]>(0, 0))[0] = 0; (a.at<uchar[3]>(0, 1))[0] = 1;
	(a.at<uchar[3]>(0, 0))[1] = 0; (a.at<uchar[3]>(0, 1))[1] = 1;
	(a.at<uchar[3]>(0, 0))[2] = 0; (a.at<uchar[3]>(0, 1))[2] = 1;
	(a.at<uchar[3]>(1, 0))[0] = 2; (a.at<uchar[3]>(1, 1))[0] = 3;
	(a.at<uchar[3]>(1, 0))[1] = 2; (a.at<uchar[3]>(1, 1))[1] = 3;
	(a.at<uchar[3]>(1, 0))[2] = 2; (a.at<uchar[3]>(1, 1))[2] = 3;

	b.at<uchar[3]>(0, 0)[0] = 4; b.at<uchar[3]>(0, 1)[0] = 5;
	b.at<uchar[3]>(0, 0)[1] = 4; b.at<uchar[3]>(0, 1)[1] = 5;
	b.at<uchar[3]>(0, 0)[2] = 4; b.at<uchar[3]>(0, 1)[2] = 5;
	b.at<uchar[3]>(1, 0)[0] = 6; b.at<uchar[3]>(1, 1)[0] = 7;
	b.at<uchar[3]>(1, 0)[1] = 6; b.at<uchar[3]>(1, 1)[1] = 7;
	b.at<uchar[3]>(1, 0)[2] = 6; b.at<uchar[3]>(1, 1)[2] = 7;

	c.at<uchar[3]>(0, 0)[0] = 8; c.at<uchar[3]>(0, 1)[0] = 9;
	c.at<uchar[3]>(0, 0)[1] = 8; c.at<uchar[3]>(0, 1)[1] = 9;
	c.at<uchar[3]>(0, 0)[2] = 8; c.at<uchar[3]>(0, 1)[2] = 9;
	c.at<uchar[3]>(1, 0)[0] = 10; c.at<uchar[3]>(1, 1)[0] = 11;
	c.at<uchar[3]>(1, 0)[1] = 10; c.at<uchar[3]>(1, 1)[1] = 11;
	c.at<uchar[3]>(1, 0)[2] = 10; c.at<uchar[3]>(1, 1)[2] = 11;

	cv::Mat merged;
	cv::merge(std::vector<cv::Mat>({a, b, c }), merged);

	for (int y = 0; y < merged.rows; ++y)
	{
		for (int x = 0; x < merged.cols; ++x)
		{
			std::cout << "{ ";
			std::cout << std::to_string(merged.at<uchar[9]>(y, x)[0]) << " ";
			std::cout << std::to_string(merged.at<uchar[9]>(y, x)[1]) << " ";
			std::cout << std::to_string(merged.at<uchar[9]>(y, x)[2]) << " ";
			std::cout << std::to_string(merged.at<uchar[9]>(y, x)[3]) << " ";
			std::cout << std::to_string(merged.at<uchar[9]>(y, x)[4]) << " ";
			std::cout << std::to_string(merged.at<uchar[9]>(y, x)[5]) << " ";
			std::cout << std::to_string(merged.at<uchar[9]>(y, x)[6]) << " ";
			std::cout << std::to_string(merged.at<uchar[9]>(y, x)[7]) << " ";
			std::cout << std::to_string(merged.at<uchar[9]>(y, x)[8]) << "} ";
		}
		std::cout << "\n";
	}
}

int main()
{
	one_channel_test();
	multi_channel_test();
}