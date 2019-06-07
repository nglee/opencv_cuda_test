#include <opencv2/opencv.hpp>

#include <iostream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

int main()
{
    const std::string inDir = "C:/Users/lee.namgoo/Documents/SuaKIT/Tutorial Projects/test/Image";
    const std::string outDir = "C:/Users/lee.namgoo/Documents/SuaKIT/Tutorial Projects/test/Image224224";

    for (const auto& p : fs::directory_iterator(inDir))
    {
        const std::string fullPath = p.path().string();                                    // inDir ���� �� directory_entry ���� ��ü �н�
        const std::string filename = p.path().filename().string();                         // ���� �̸�
        const std::string extension = p.path().extension().string();                       // Ȯ���� ('.' ����)

        if (extension != ".jpg")
            continue;

		const std::string new_fullPath = outDir + "/" + filename;

        const cv::Mat img = cv::imread(fullPath, cv::IMREAD_COLOR);
		cv::Mat new_img;

		cv::resize(img, new_img, cv::Size(224, 224));

        cv::imwrite(new_fullPath, new_img);

        std::cout << new_fullPath << std::endl;
    }
}