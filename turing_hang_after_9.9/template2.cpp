#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv) {

    cv::cuda::setDevice(0); // initialize CUDA
    cv::Mat result_h;   
    //cv::Mat image_h = cv::imread(argv[1]); // input
    //cv::Mat templ_h = cv::imread(argv[2]); // template 
    cv::Mat image_h(10293, 10900, CV_8UC3);
    cv::Mat templ_h(289, 10, CV_8UC3);

    // convert from mat to gpumat
    cv::cuda::GpuMat image_d(image_h);
    cv::cuda::GpuMat templ_d(templ_h);
    cv::cuda::GpuMat result;

    // GPU -> NG
    std::cout << "debug msg 1" << std::endl;
    cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(image_h.type(), cv::TM_CCOEFF_NORMED);
    std::cout << "debug msg 2" << std::endl;
    alg->match(image_d, templ_d, result);  // no return.
    std::cout << "debug msg 3" << std::endl;

    cv::cuda::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1);
    double max_value;
    cv::Point location;
    cv::cuda::minMaxLoc(result, 0, &max_value, 0, &location);

/*
    // CPU -> OK
    cv::matchTemplate(image_h, templ_h, result_h, cv::TM_CCOEFF_NORMED);
    cv::normalize(result_h, result_h, 0, 1, cv::NORM_MINMAX, -1);
    double max_value;
    cv::Point location;
    cv::minMaxLoc(result_h, 0, &max_value, 0, &location);
*/
    std::cout << "======Test Match Template======" << std::endl;
    std::cout << "input :" << argv[1] << std::endl;
    std::cout << "template :" << argv[2] << std::endl;
    std::cout << " " << std::endl;

    return 0;
}
