# build file for a mediapipe c++ library

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

# external shared library
cc_binary(
    name = "mediapipe",
    srcs = ["face_landmarker.cc", "mediapipe.hh", "mediapipe_detail.hh"],
    deps = [
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:classification_result",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_result",
    ],
    linkshared = True,
)

cc_test(
    name = "mediapipe_test",
    srcs = ["mediapipe_test.cc"],
    deps = [
        ":mediapipe_lib",
        "//mediapipe/framework/port:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "mediapipe_main",
    srcs = ["face_landmarker_ccl_main.cc"],
    deps = [
        ":mediapipe_lib",
  
        # load image
        "//mediapipe/framework/port:opencv_imgcodecs",
        # cv::VideoCapture
        "//mediapipe/framework/port:opencv_video",
        # cv::imshow
        "//mediapipe/framework/port:opencv_highgui",
    ],
)

# internal for bazel
cc_library(
    name = "mediapipe_lib",
    srcs = ["face_landmarker.cc"],
    hdrs = ["mediapipe.hh", "mediapipe_detail.hh"],
    deps = [
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:classification_result",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_result",
    ],
)
