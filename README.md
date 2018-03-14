## Environment
- OS: Windows 10
- IDE: Visual Studio 2015

## Windows Environment Variable Settings
- _$(OPENCV_2_4_13_2)_ : set to the install directory of `OpenCV 2.4.13.2`
- _$(OPENCV_3_1)_ : set to the install directory of `OpenCV 3.1`
- _$(OPENCV_3_2)_ : set to the install directory of `OpenCV 3.2`
- _$(OPENCV_3_4)_ : set to the install directory of `OpenCV 3.4`
- _$(OPENCV_DEV)_ : set to the install directory of `OpenCV` master branch
- _$(PATH)_ : add paths to `.dll` files of each `OpenCV` version

## Note
For OpenCV version `2.4.13.2`, prebuilt binaries are used that are downloadable from the OpenCV official website.

For OpenCV version `3.1`, QT backend was enabled during customary build process. Extra modules are also included.
Among the extra modules, `opencv_bioinspired` and `opencv_cvv` were not selected since it had compiler errors on my machine, and I didn't need them anyway.

When building `3.1`, a patch described [here](https://github.com/opencv/opencv/issues/6677) were applied to make it compatible with `CUDA 8.0`. To apply the patch, type the following command:

```
$ git format-patch -1 10896129b39655e19e4e7c529153cb5c2191a1db
$ git am < 0001-GraphCut-deprecated-in-CUDA-7.5-and-removed-in-8.0.patch
```

For OpenCV version `3.2`, extra modules are also included and only `opencv_cvv` were disabled.
Do not forget to enable `OPENCV_ENABLE_NONFREE` when configuring `3.2` to use `SIFT` and `SURF`.

Sample images are from [pixabay](https://pixabay.com/).
