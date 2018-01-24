#include <opencv2/opencv.hpp>

int main()
{
	const std::string inputPath = "C:/Users/lee.namgoo/Desktop/input.jpg";
	const std::string outputPath = "C:/Users/lee.namgoo/Desktop/output.jpg";

	cv::Mat img = cv::imread(inputPath);
	cv::Mat img_out;

	if (img.channels() == 3) {
		std::cout << "converting 3 channel image to 1 channel image\n";
		cv::cvtColor(img, img_out, CV_BGR2GRAY);
		cv::imwrite(outputPath, img_out);
	} else {
		std::cout << "input image is not a 3 channel image\n";
	}

	cv::Mat img_luv;
	if (img.channels() == 3) {
		cv::cvtColor(img, img_luv, CV_BGR2YUV);
		cv::imwrite("C:/Users/lee.namgoo/Desktop/Yuv.png", img_luv);

		cv::cuda::GpuMat img_luv_d, img_bgr_d;
		img_luv_d.upload(img_luv);

		cv::cuda::cvtColor(img_luv_d, img_bgr_d, CV_YUV2BGR);

		img_bgr_d.download(img_out);
		cv::imwrite("C:/Users/lee.namgoo/Desktop/BGR_d.png", img_out);

		cv::cvtColor(img_luv, img_out, CV_YUV2BGR);
		cv::imwrite("C:/Users/lee.namgoo/Desktop/BGR_h.png", img_out);
	}

}