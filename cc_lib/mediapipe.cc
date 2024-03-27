// i now try to mimic the python program wrote in c++

// bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_bin && ./bazel-mediapipe/bazel-out/darwin-opt/bin/cc_lib/mediapipe_bin

#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"

// cv::imread
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"

#include <iostream>

using namespace std;

using mediapipe::tasks::vision::core::RunningMode;
using mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions;
using mediapipe::tasks::vision::face_landmarker::FaceLandmarker;

using mediapipe::ImageFormat;
using mediapipe::ImageFrame;
using mediapipe::Image;

// mediapipe_cpp_lib was able to use an external cv and protobuf library

int main() {
    cout << "start" << endl;

    auto cpp_options = std::make_unique<FaceLandmarkerOptions>();
    cpp_options->base_options.model_asset_path = "/Users/mark/python/py311-venv-mediapipe/face_landmarker_v2_with_blendshapes.task";
    cpp_options->running_mode = RunningMode::IMAGE;
    cpp_options->output_face_blendshapes = true;
    cpp_options->output_facial_transformation_matrixes = true;
    cpp_options->num_faces = 1;

    auto landmarker = FaceLandmarker::Create(std::move(cpp_options));
    if (!landmarker.ok()) {
        cerr << "Failed to create FaceLandmarker: " << landmarker.status() << endl;
        return 1;
    }

    cv::Mat input = cv::imread("mediapipe/objc/testdata/sergey.png");
    if (input.data == nullptr) {
        cerr << "failed to load image" << endl;
        return 1;
    }
    ImageFrame image_frame(
        input.channels() == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB,
        input.cols, input.rows, input.step, input.data, [](uint8_t *) {});
    Image image(std::make_shared<mediapipe::ImageFrame>(std::move(image_frame)));

    auto result = (*landmarker)->Detect(image);
    if (!result.ok()) {
        cerr << "Detection failed: " << result.status() << endl;
        return 1;
    }

    for (uint32_t i = 0; i < result->face_landmarks.size(); ++i) {
        cout << "landmark[" << i << "].size() = " << result->face_landmarks[i].landmarks.size() << endl;
    }
    if (result->face_blendshapes.has_value()) {
        cout << "we have blend shapes" << endl;
    }
    if (result->facial_transformation_matrixes.has_value()) {
        cout << "we have facial_transformation_matrixes" << endl;
    }

    cout << "finish" << endl;

    return 0;
}
