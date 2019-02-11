#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    cuda::GpuMat in(787, 1987, CV_8UC1);
    cuda::GpuMat out;

    Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create("haarcascade_eye.xml");

    cascade_gpu->detectMultiScale(in, out);

    return 0;
}
