#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <mutex>

	cv::Mat imread_w(const std::wstring& filename, int flags)
	{
#ifdef _WIN32
		FILE* fp = _wfopen(filename.c_str(), L"rb");
#else
		FILE* fp = fopen(std::string(filename.begin(), filename.end()).c_str(), "rb");
#endif
		fseek(fp, 0, SEEK_END);
		long sz = ftell(fp);
		static thread_local std::vector<char> t_buffer; // 반복적인 할당, 해제를 줄이기 위하여..
		t_buffer.resize(sz);
		fseek(fp, 0, SEEK_SET);
		long n = fread(t_buffer.data(), 1, sz, fp);
		cv::_InputArray arr(t_buffer.data(), sz);
		
		// https://sualab.atlassian.net/browse/SKP-1
		// some opencv APIs are not thread-safe, so embrace cv::imdecode with mutex to get it threa-safe forcily
		cv::Mat img;
		static std::mutex mtx;
		{
			std::lock_guard<std::mutex> mmtx{ mtx };
			img = cv::imdecode(arr, flags);
		}
		fclose(fp);

		return img;
	}

int main()
{
    std::vector<std::wstring> imgs = {
        L"E:/SuaKIT down images/SuaKIT down images/IMG_1147.JPG",
        L"E:/SuaKIT down images/SuaKIT down images/IMG_1148.JPG",
        L"E:/SuaKIT down images/SuaKIT down images/IMG_1149.JPG",
        L"E:/SuaKIT down images/SuaKIT down images/IMG_1150.JPG",
        L"E:/SuaKIT down images/SuaKIT down images/IMG_1151.JPG",
        L"E:/SuaKIT down images/SuaKIT down images/IMG_1152.JPG"
	};

	for (const auto& path : imgs) {
		//cv::Mat mat = cv::imread(path, CV_LOAD_IMAGE_COLOR);
		//cv::imshow("temp", mat);

		//std::cout << "cols : " << mat.cols << " rows : " << mat.rows << "\n";

		//cv::waitKey();

		cv::Mat mat = imread_w(path, CV_LOAD_IMAGE_COLOR);
		std::cout << "cols : " << mat.cols << " rows : " << mat.rows << "\n";
	}
}