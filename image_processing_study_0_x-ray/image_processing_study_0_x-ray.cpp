#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

class TestData
{
public:
	TestData(const std::string& path_str, size_t _w, size_t _h, size_t _cv_depth)
		: path(path_str), w(_w), h(_h), cv_depth(_cv_depth)
	{}

	~TestData() = default;

	TestData(const TestData& rhs) = delete;
	TestData& operator=(const TestData& rhs) = delete;
	TestData& operator=(TestData&& rhs) = delete;

	TestData(TestData&& rhs) noexcept
		: path(std::move(rhs.path))
	{
		w = rhs.w;
		h = rhs.h;
		cv_depth = rhs.cv_depth;
	}
	
	fs::path path;
	size_t w;
	size_t h;
	size_t cv_depth;
};

void do_test(const TestData&);

int main()
{
	std::array<TestData, 3> tds = {
		TestData( "../resource/belly_3072x3072_16bit.raw", 3072, 3072, CV_16U ),
		TestData( "../resource/chest_3072x3072_16bit.raw", 3072, 3072, CV_16U ),
		TestData( "../resource/spine_3072x3072_16bit.raw", 3072, 3072, CV_16U )
	};

	try {
		for (const auto& td : tds)
		{
			do_test(td);
		}
	}
	catch (std::exception& e)
	{
		std::cout << "Exception in do_test: " << e.what() << std::endl;
	}
}

void do_test(const TestData& td)
{
	std::ifstream ifs(td.path.c_str(), std::ios::binary);

	std::vector<ushort> _data(td.w * td.h);

	ifs.read(static_cast<char*>(static_cast<void*>(_data.data())), _data.size() * sizeof(ushort));

	const cv::Mat input(td.h, td.w, CV_16UC1, _data.data());

	//const auto rotate = std::make_unique<cv::Mat>();
	cv::Mat rotate;
	cv::rotate(~input, rotate, cv::ROTATE_90_COUNTERCLOCKWISE);

	cv::Mat hist;
	int channels[] = { 0 };
	int hist_size[] = { 256 };
	float ranges[][2] = { { 0.0f, 65535.0f } };
	cv::calcHist(&rotate, 1, channels, cv::Mat(), hist, 1, hist_size, ranges);
}