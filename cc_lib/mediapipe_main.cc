#include "mediapipe.hh"

// cv::imread
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
// cv::VideoCapture

#include "mediapipe/framework/port/opencv_video_inc.h"

// https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
#include "mediapipe/framework/port/opencv_highgui_inc.h"

#include <iostream>

using mediapipe::cc_lib::vision::core::RunningMode;
using mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerOptions;
using mediapipe::cc_lib::vision::face_landmarker::FaceLandmarker;

using namespace std;

// int main() {
//   cout << "start" << endl;

//   auto cpp_options = std::make_unique<FaceLandmarkerOptions>();
//   cpp_options->base_options.model_asset_path =
//       "/Users/mark/python/py311-venv-mediapipe/"
//       "face_landmarker_v2_with_blendshapes.task";
//   cpp_options->running_mode = RunningMode::LIVE_STREAM;
//   cpp_options->output_face_blendshapes = true;
//   cpp_options->output_facial_transformation_matrixes = true;
//   cpp_options->num_faces = 1;

//   cpp_options->result_callback =
//       [&](absl::StatusOr<FaceLandmarkerResult> result, const Image &image,
//           int64_t timestamp_ms) {
//         // this might be called from a thread hence imgshow won't work here
//         // (macOS needs to run opencv in the main thread)
//         cout << "callback" << endl;
//       };

//   auto landmarker = FaceLandmarker::Create(std::move(cpp_options));
//   if (!landmarker.ok()) {
//     cerr << "Failed to create FaceLandmarker: " << landmarker.status() << endl;
//     return 1;
//   }

//   cv::VideoCapture cap(0);
//   if (!cap.isOpened()) {
//     cerr << "failed to open video" << endl;
//     return 1;
//   }
//   cv::Mat videoFrameCV;
//   while (true) {
//     cap >> videoFrameCV;
//     if (videoFrameCV.empty()) {
//       std::cout << "empty image" << std::endl;
//       return 1;
//     }

//     ImageFrame imageFrameMP(
//         videoFrameCV.channels() == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB,
//         videoFrameCV.cols, videoFrameCV.rows, videoFrameCV.step,
//         videoFrameCV.data, [](uint8_t *) {});
//     Image imageMP(
//         std::make_shared<mediapipe::ImageFrame>(std::move(imageFrameMP)));

//     struct timeval tv;
//     gettimeofday(&tv, NULL);
//     unsigned long long timestamp = (unsigned long long)(tv.tv_sec) * 1000 +
//                                    (unsigned long long)(tv.tv_usec) / 1000;

//     auto status = (*landmarker)->DetectAsync(imageMP, timestamp);
//     if (!status.ok()) {
//       cerr << "Detection failed: " << status << endl;
//       return 1;
//     }

//     imshow("Display window", videoFrameCV);
//     if (waitKey(30) >= 0) {
//       break;
//     }
//   }
//   return 0;
// }

int main() {
  cout << "start" << endl;

  auto cpp_options = std::make_unique<FaceLandmarkerOptions>();
  cpp_options->base_options.model_asset_path =
      "/Users/mark/python/py311-venv-mediapipe/"
      "face_landmarker_v2_with_blendshapes.task";
  cpp_options->running_mode = RunningMode::IMAGE;
  cpp_options->output_face_blendshapes = true;
  cpp_options->output_facial_transformation_matrixes = true;
  cpp_options->num_faces = 1;

  auto landmarker = FaceLandmarker::Create(std::move(cpp_options));
//   if (!landmarker.has_value()) {
//     cerr << "Failed to create FaceLandmarker: " << landmarker.error().what() << endl;
//     return 1;
//   }
    if (!landmarker) {
        cerr << "Failed to create FaceLandmarker"  << endl;
        return 1;
    }

    // https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8
    cv::Mat input = cv::imread("mediapipe/objc/testdata/sergey.png");
    if (input.data == nullptr) {
        cerr << "failed to load image" << endl;
        return 1;
    }

    landmarker->Detect(input.channels(), input.cols, input.rows, input.step, input.data);

//   auto result = (*landmarker)->Detect(image);
//   if (!result.ok()) {
//     cerr << "Detection failed: " << result.status() << endl;
//     return 1;
//   }

//   for (uint32_t face = 0; face < result->face_landmarks.size(); ++face) {
//     cout << "landmark[" << face
//          << "].size() = " << result->face_landmarks[face].landmarks.size()
//          << endl;
//   }
//   if (result->face_blendshapes.has_value()) {
//     cout << "we have blend shapes" << endl;
//   }
//   if (result->facial_transformation_matrixes.has_value()) {
//     cout << "we have facial_transformation_matrixes" << endl;
//   }

//   cv::namedWindow("edges", cv::WINDOW_NORMAL); // create window named 'edges'
//   cv::imshow("edges", input);
//   cv::waitKey(0);

  cout << "finish" << endl;

  return 0;
}
