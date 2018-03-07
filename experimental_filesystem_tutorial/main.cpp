#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <experimental/filesystem>

int main()
{
    namespace fs = std::experimental::filesystem;

    const std::string inDir = "Input/Directory/Path";
    const std::string outDir = "Output/Directory/Path";
    int count = 0;
                        
    for (const auto& p : fs::directory_iterator(inDir))
    {
        const std::string fullPath = p.path().string();                                    // inDir 이하 각 directory_entry 들의 전체 패스
        const std::string grandparentPath = p.path().parent_path().parent_path().string(); // parent_path 연속해서 두 번 먹임
        const std::string filename = p.path().filename().string();                         // 파일 이름
        const std::string stem = p.path().stem().string();                                 // 확장자를 제외한 파일 이름
        const std::string extension = p.path().extension().string();                       // 확장자 ('.' 포함)

        if (extension != ".png")
            continue;
        
        const int token = stem.find('_');
        const int mainNum = atoi(stem.substr(0, token).c_str());
        const int minorNum = atoi(stem.substr(token + 1).c_str());

        std::stringstream new_mainNum_stream;
        new_mainNum_stream << std::setw(5) << std::setfill('0') << mainNum;

        std::string new_filename;
        switch (minorNum)
        {
        case 0:
            new_filename = std::string(new_mainNum_stream.str() + "_defect.png");
            break;
        case 1:
            new_filename = std::string(new_mainNum_stream.str() + "_mask.png");
            break;
        case 2:
            new_filename = std::string(new_mainNum_stream.str() + "_defect_label.png");
            break;
        case 3:
            new_filename = std::string(new_mainNum_stream.str() + "_replica.png");
            break;
        }

        const std::string new_fullPath = outDir + new_filename;

        const cv::Mat img = cv::imread(fullPath, cv::IMREAD_GRAYSCALE);
        cv::imwrite(new_fullPath, img);

        std::cout << new_fullPath << std::endl;
    }
}