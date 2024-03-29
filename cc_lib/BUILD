# build file for a mediapipe c++ library

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "mediapipe_main",
    srcs = ["mediapipe_main.cc"],
    deps = [
        ":mediapipe",
        
        # load image
        "//mediapipe/framework/port:opencv_imgcodecs",
        # cv::VideoCapture
        "//mediapipe/framework/port:opencv_video",
        # cv::imshow
        "//mediapipe/framework/port:opencv_highgui",
        # "//mediapipe/framework/port:opencv_imgproc",

        # "//mediapipe/framework/formats:image_frame",
        # "//mediapipe/framework/formats:image_frame_opencv",
    ],
)

cc_test(
    name = "mediapipe_test",
    srcs = ["mediapipe_test.cc"],
    deps = [
        ":mediapipe",
        "//mediapipe/framework/port:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_shared_library(
    name = "mediapipe_shared",
    deps = [":mediapipe"],
    shared_lib_name = "libmediapipe.so",
)

cc_library(
    name = "mediapipe",
    srcs = ["mediapipe.cc"],
    hdrs = ["mediapipe.hh", "mediapipe_detail.hh"],
    deps = [
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:classification_result",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_result",
    ],
)
