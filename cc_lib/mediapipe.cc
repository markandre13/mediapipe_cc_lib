#include "mediapipe_detail.hh"

// i now try to mimic the python program wrote in c++

// bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_bin &&
// ./bazel-mediapipe/bazel-out/darwin-opt/bin/cc_lib/mediapipe_bin

// #include "mediapipe/tasks/cc/vision/core/running_mode.h"
// #include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
// #include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"

// cv::imread
// #include "mediapipe/framework/port/opencv_imgcodecs_inc.h"

// cv::VideoCapture
// #include "mediapipe/framework/port/opencv_video_inc.h"

// https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
// #include "mediapipe/framework/port/opencv_highgui_inc.h"

// #include "mediapipe/framework/formats/image_frame.h"
// #include "mediapipe/framework/formats/image_frame_opencv.h"

// #include <iostream>
// #include <sys/time.h>

// using namespace std;

// using mediapipe::tasks::vision::core::RunningMode;
// using mediapipe::tasks::vision::face_landmarker::FaceLandmarker;
// using mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions;

// using mediapipe::Image;
// using mediapipe::ImageFormat;
// using mediapipe::ImageFrame;

// mediapipe_cpp_lib was able to use an external cv and protobuf library

// using namespace cv;

// using mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult;

namespace mediapipe {
namespace cc_lib {
namespace detail {

std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>
convert(const std::unique_ptr<mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerOptions> &in) {
    auto out = std::make_unique<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>();

    out->base_options.model_asset_path = in->base_options.model_asset_path;
    out->running_mode = static_cast<mediapipe::tasks::vision::core::RunningMode>(in->running_mode);
    out->num_faces = in->num_faces;
    out->min_face_detection_confidence = in->min_face_detection_confidence;
    out->min_face_presence_confidence = in->min_face_presence_confidence;
    out->min_tracking_confidence = in->min_tracking_confidence;
    out->output_face_blendshapes = in->output_face_blendshapes;
    out->output_facial_transformation_matrixes = in->output_facial_transformation_matrixes;

    return out;
}

} // namespace detail

namespace vision {
namespace face_landmarker {

FaceLandmarkerOptions::FaceLandmarkerOptions() {}

std::unique_ptr<FaceLandmarker>
FaceLandmarker::Create(std::unique_ptr<FaceLandmarkerOptions> options) {
    auto flm = mediapipe::tasks::vision::face_landmarker::FaceLandmarker::Create(std::move(
        mediapipe::cc_lib::detail::convert(options)
    ));
    if (!flm.ok()) {
        return {};
    }
    // std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarker> ptr(std::move(*flm));
    // new FaceLandmarker(std::move(ptr));
    auto result = std::make_unique<FaceLandmarker>();
    result->mp = std::move(*flm);
    return result;
}

FaceLandmarker::~FaceLandmarker() {}

// FaceLandmarker::FaceLandmarker(std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarker> &mp): mp(std::move(mp)) {}

} // namespace face_landmarker
} // namespace vision
} // namespace cc_lib
} // namespace mediapipe
