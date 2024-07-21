/**
 * Use Mediapipe's native C++ API to analyze a live video stream
 */

#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"

// cv::imread
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"

// cv::VideoCapture
#include "mediapipe/framework/port/opencv_video_inc.h"

// https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
#include "mediapipe/framework/port/opencv_highgui_inc.h"

#include <iostream>

using cv::waitKey;
using mediapipe::Image;
using mediapipe::ImageFormat;
using mediapipe::ImageFrame;
using mediapipe::tasks::vision::core::RunningMode;
using mediapipe::tasks::vision::face_landmarker::FaceLandmarker;
using mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions;
using mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult;

using namespace std;

int face_landmarker_native() {
  cout << "start face_landmarker_native" << endl;

  auto cpp_options = std::make_unique<FaceLandmarkerOptions>();
  cpp_options->base_options.model_asset_path = "cc_lib/face_landmarker.task";
  cpp_options->running_mode = RunningMode::LIVE_STREAM;
  cpp_options->output_face_blendshapes = true;
  cpp_options->output_facial_transformation_matrixes = true;
  cpp_options->num_faces = 1;

  cpp_options->result_callback =
      [&](absl::StatusOr<FaceLandmarkerResult> result, const Image &image,
          int64_t timestamp_ms) {
        // this might be called from a thread hence imgshow won't work here
        // (macOS needs to run opencv in the main thread)
        cout << "callback" << endl;
        if (!result.ok()) {
          cout << "  result not ok" << endl;
        } else {
          cout << "  number of faces: " << result->face_landmarks.size()
               << endl;
          for (auto &face : result->face_landmarks) {
            cout << "    face with " << face.landmarks.size() << " landmarkers"
                 << endl;
            if (result->face_blendshapes) {
              cout << "    have blendshapes" << endl;
            }
            if (result->facial_transformation_matrixes) {
              cout << "    have facial_transformation_matrixes" << endl;
            }
          }
        }
      };

  auto landmarker = FaceLandmarker::Create(std::move(cpp_options));
  if (!landmarker.ok()) {
    cerr << "Failed to create FaceLandmarker: " << landmarker.status() << endl;
    return 1;
  }

  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    cerr << "failed to open video" << endl;
    return 1;
  }
  cv::Mat videoFrameCV;
  while (true) {
    cap >> videoFrameCV;
    if (videoFrameCV.empty()) {
      std::cout << "empty image" << std::endl;
      return 1;
    }

    ImageFrame imageFrameMP(
        videoFrameCV.channels() == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB,
        videoFrameCV.cols, videoFrameCV.rows, videoFrameCV.step,
        videoFrameCV.data, [](uint8_t *) {});
    Image imageMP(
        std::make_shared<mediapipe::ImageFrame>(std::move(imageFrameMP)));

    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long timestamp = (unsigned long long)(tv.tv_sec) * 1000 +
                                   (unsigned long long)(tv.tv_usec) / 1000;

    auto status = (*landmarker)->DetectAsync(imageMP, timestamp);
    if (!status.ok()) {
      cerr << "Detection failed: " << status << endl;
      return 1;
    }

    imshow("Display window", videoFrameCV);
    if (waitKey(30) >= 0) {
      break;
    }
  }
  return 0;
}
