#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <Windows.h>
void FilePathsInDir(const std::string dir, std::vector<std::string>& paths)
{
	char searchPath[_MAX_PATH];
	sprintf(searchPath, "%s*.*", dir.c_str());
	WIN32_FIND_DATA fd;
	HANDLE hFind = FindFirstFile(searchPath, &fd);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		std::cerr << "FindFirstFile failed\n";
	}
	else
	{
		do {
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				std::string tmpStr = fd.cFileName;
				
				if (std::string::npos == tmpStr.find("bin"))
					continue;

				paths.push_back(dir + tmpStr);
			}
		} while (FindNextFile(hFind, &fd));
		FindClose(hFind);
	}

	std::sort(paths.begin(), paths.end());

	return;
}

std::string CreateOutputPath(std::string dir, int number)
{
	char buf[60];
	sprintf(buf, "%soutput/%04d.png", dir.c_str(), number);
	return std::string(buf);
}

int main(int argc, const char* argv[])
{
	const std::string dir = "D:/image_tests/blob_detector_test_images/";
	std::vector<std::string> paths;
	FilePathsInDir(dir, paths);

	int number = -1;
	for (std::string path : paths) {
		number++;
		cv::Mat input = cv::imread(path, 0), input_resized;
		cv::resize(input, input_resized, cv::Size(), 0.9, 0.9);

		const std::string output_path = CreateOutputPath(dir, number);

		std::vector<cv::KeyPoint> keypoints;
		cv::SimpleBlobDetector::Params param;
		param.filterByArea = 0;
		param.filterByCircularity = 0;
		param.filterByColor = 0;
		param.filterByConvexity = 0;
		param.filterByInertia = 0;
		cv::Ptr<cv::SimpleBlobDetector> sbd = cv::SimpleBlobDetector::create(param);
		sbd->detect(input_resized, keypoints);

		cv::Mat output;
		cv::drawKeypoints(input_resized, keypoints, output, cv::Scalar_<double>::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cv::imwrite(output_path, output);
	}
	return 0;
}