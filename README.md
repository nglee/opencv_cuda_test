## Environment
- OS: Windows 10
- IDE: Visual Studio 2015
- Compiler: vc14

## Windows Environment Variable Settings
- _$(OPENCV_2_4_13_2)_ : set to the install directory of `OpenCV 2.4.13.2`
- _$(OPENCV_3_1)_ : set to the install directory of `OpenCV 3.1`
- _$(OPENCV_3_2)_ : set to the install directory of `OpenCV 3.2`
- _$(OPENCV_3_4)_ : set to the install directory of `OpenCV 3.4`
- _$(OPENCV_DEV)_ : set to the install directory of `OpenCV` master branch
- _$(PATH)_ : add paths to `.dll` files of each `OpenCV` version

## Note
When doing this project, I've used prebuilt binaries for `2.4.13.2`, downloadable from the `OpenCV` website. However, for `3.1` and `3.2`, I've built a customary version for myself to include extra modules.

Among the extra modules, `opencv_bioinspired` and `opencv_cvv` were not selected for `3.1` since it had compiler errors on my machine, and I didn't need them anyway. For `3.2`, only `opencv_cvv` were disabled.

When building `3.1`, a patch described [here](https://github.com/opencv/opencv/issues/6677) were applied to make it compatible with `CUDA8.0`. To apply the patch, type following command:

```
$ git format-patch -1 10896129b39655e19e4e7c529153cb5c2191a1db
$ git am < 0001-GraphCut-deprecated-in-CUDA-7.5-and-removed-in-8.0.patch
```

Do not forget to enable `OPENCV_ENABLE_NONFREE` when configuring `3.2` to use `SIFT` and `SURF`.
