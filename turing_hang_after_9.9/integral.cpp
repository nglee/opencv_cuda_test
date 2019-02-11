#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

void cudaIntegral(InputArray m1, OutputArray pts) {
    GpuMat gm1, gm2;
    gm1.upload(m1);
    CV_Assert(!gm1.empty());

    cv::cuda::integral(gm1, gm2);

    gm2.download(pts);
}

int main(int argc, char **argv) {
    //Mat in = imread(argv[1], IMREAD_GRAYSCALE);
    Mat in(787, 1987, CV_8UC1);
    Mat *out = new Mat();

    cudaIntegral(in, *out);

    cout << *out << endl;
    delete out;
    return 0;
}
