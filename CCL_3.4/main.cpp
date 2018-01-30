#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

void test(const std::string& in_path = "", const float w2b_ratio = 0.5f)
{
	std::cout << "Testing with : " << in_path << "\n";

	cv::Mat in;
	cv::RNG rng;

	if (in_path == "") {
		int rows = 2048, cols = 2048;

		in = cv::Mat{ rows, cols, CV_8UC1, cv::Scalar(0) };

		cv::Mat tmp{ rows, cols, CV_32FC1, cv::Scalar(0.0f) };

		for (int r = 0; r < rows; r++)
			for (int c = 0; c < cols; c++)
				tmp.at<float>(r, c) = rng.uniform(0.0f, 1.0f);

		in = (tmp < w2b_ratio);
	} else {
		in = cv::imread(in_path, cv::IMREAD_GRAYSCALE);
	}
	//cv::imshow("tmp", in);
	//cv::waitKey();

	cv::Mat labels, stats, centroids;

	// Dummy operation for benchmark
	cv::connectedComponents(in, labels);

	// Format output for benchmark results
	std::cout << std::setprecision(6) << std::fixed;

	auto clk_now = std::chrono::high_resolution_clock::now();
	int N = cv::connectedComponents(in, labels);
	std::cout << std::setw(40) << std::left << "connectedComponents" << ": " << std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - clk_now).count() << " ms\n";
	cv::imwrite(in_path + ".labels1.bmp", labels);

	clk_now = std::chrono::high_resolution_clock::now();
	N = cv::connectedComponentsWithStats(in, labels, stats, centroids);
	std::cout << std::setw(40) << std::left << "connectedComponentsWithStats" << ": " << std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - clk_now).count() << " ms\n";
	cv::imwrite(in_path + ".labels2.bmp", labels);

	std::cout << std::endl;
}

int main()
{
	test("./bin_512_512.bmp");
	test("./bin_1024_1024.bmp");
	test("./bin_2048_2048.bmp");
	test("", 0.10f);
	test("", 0.20f);
	test("", 0.30f);
	test("", 0.40f);
	test("", 0.50f);
	test("", 0.60f);
	test("", 0.70f);
	test("", 0.80f);
	test("", 0.90f);
}