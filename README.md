## Environment
OS: Windows 10
IDE: Visual Studio 2015
Compiler: vc14

## Windows Environment Variable Settings
OPENCV_2_4_13_2 : install directory of OpenCV 2.4.13.2
OPENCV_3_1 : install directory of OpenCV 3.1
OPENCV_3_2 : install directory of OpenCV 3.2
PATH : paths to .dll files of each OpenCV version

## Remark
When doing this project, I've used prebuilt binaries for 2.4.13.2, downloadable from the OpenCV website. However, for 3.1 and 3.2, I've built a customary version for myself to include extra modules.

Among the extra modules, opencv_bioinspired and opencv_cvv were not selected for 3.1 since it had compiler errors on my machine, and I didn't need them anyway. For 3.2, only opencv_cvv were disabled.

When building 3.1, a patch described [here](https://github.com/opencv/opencv/issues/6677) were applied to make it compatible with CUDA8.0. Specific commands are:

```
$ git format-patch -1 10896129b39655e19e4e7c529153cb5c2191a1db
$ git am < 0001-GraphCut-deprecated-in-CUDA-7.5-and-removed-in-8.0.patch
```