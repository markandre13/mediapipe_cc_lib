# Mediapipe Standalone C++ Library

## introduction

while mediapipe is written in c++, it does not build a c++ library. (but there are an
internal c api, and external python, java, javascript and objc api's.)

[mediapipe_cpp_lib](https://github.com/purgeme/mediapipe_cpp_lib) solved that but
on 2024-02-19 was put on pause due to a new api in mediapipe using 'tasks'.

therefore i had to come up with something on my own.

* only macos (too early to worry about cross-platform)
* only face_landmarker (that's what i needed 1st)
* api is close to mediapipe's internal api's
* mediapipe_cc_lib has it's own headers as mediapipe's internal ones contain too many dependencies

## mediapipe

### how mediapipe builds

* builds with bazel, which is google's internal build tool. it's written in java, can download additional files, etc.
* absl: google's c++ utility library
* tensorflow (lite): google's neural network library
* protocol buffers: google's programming language independent
data exchange format
* details on the [face landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker): task file and detailed description are linked at the bottom of the page

## build

### opencv

the [installation guide](https://developers.google.com/mediapipe/framework/getting_started/install) states

> To interoperate with OpenCV, OpenCV 3.x to 4.1 are preferred.

macOS brew is on 4.9 and the build fails with

    ./mediapipe/framework/port/opencv_core_inc.h:18:10: fatal error: 'opencv2/core/version.hpp' file not found

opencv3 works but has been disabled on 2024-01-31. it can be still installed with

    HOMEBREW_NO_INSTALL_FROM_API=1 brew install opencv@3
    brew edit opencv@3

remove the line `disable! date: "2024-01-31", because: :unmaintained` then run again

    HOMEBREW_NO_INSTALL_FROM_API=1 brew install opencv@3

### build

    cd cc_lib && make

### test

    cd cc_lib && make test

### develop

to try things out, put a main() into `cc_lib/mediapipe.cc` and run

    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_main && ./bazel-mediapipe_cc_lib/bazel-out/darwin-opt/bin/cc_lib/mediapipe_main

### test

    bazel test --test_output=all --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_test
