# build file for a mediapipe c++ library

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "mediapipe_bin",
    deps = [":mediapipe"],
)

cc_shared_library(
    name = "mediapipe_shared",
    deps = [":mediapipe"],
    shared_lib_name = "libmediapipe.so",
)

cc_library(
    name = "mediapipe",
    srcs = ["mediapipe.cc"],
    hdrs = ["mediapipe.hh"],
    deps = [
        "//mediapipe/tasks/cc/components/containers:category",
        "//mediapipe/tasks/cc/components/containers:classification_result",
        "//mediapipe/tasks/cc/components/containers:landmark",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker",
        "//mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_result",
        "//mediapipe/framework/port:opencv_imgcodecs",
        # cv::VideoCapture
        "//mediapipe/framework/port:opencv_video",
        # cv::imshow
        "//mediapipe/framework/port:opencv_highgui",

        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        # "//mediapipe/framework/port:opencv_imgproc",
    ],
)
