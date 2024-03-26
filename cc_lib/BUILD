# build file for a mediapipe c++ library

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_shared_library(
    name = "mediapipe_shared",
    deps = [":mediapipe"],
    shared_lib_name = "libmediapipe.so",
    # linkshared = True,
)

cc_library(
    name = "mediapipe",
    srcs = ["mediapipe.cc"],
    # linkshared = True,
)
