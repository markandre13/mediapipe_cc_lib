/**
 * Use Mediapipe's native C++ API to analyze a live video stream
 */

#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"

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
using mediapipe::tasks::vision::pose_landmarker::PoseLandmarker;
using mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerOptions;
using mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult;

using namespace std;

int main() {
  cout << "start pose_landmarker_native" << endl;

  auto cpp_options = std::make_unique<PoseLandmarkerOptions>();
  cpp_options->base_options.model_asset_path = "cc_lib/pose_landmarker_full.task";
  cpp_options->running_mode = RunningMode::LIVE_STREAM;
  cpp_options->num_poses = 1;
  cpp_options->min_pose_detection_confidence = 0.5;
  cpp_options->min_pose_presence_confidence = 0.5;
  cpp_options->min_tracking_confidence = 0.5;
  cpp_options->output_segmentation_masks = false;

  cpp_options->result_callback =
      [&](absl::StatusOr<PoseLandmarkerResult> result, const Image &image,
          int64_t timestamp_ms) {
        // this might be called from a thread hence imgshow won't work here
        // (macOS needs to run opencv in the main thread)
        cout << "callback" << endl;
        if (!result.ok()) {
          cout << "  result not ok" << endl;
        } else {
            cout << "  have segmentation mask: " << (result->segmentation_masks.has_value() ? "yes" : "no") << endl;
            cout << "  poses                 : " << result->pose_landmarks.size() << endl;
            for(auto &pose: result->pose_landmarks) {
                cout << "    pose has " << pose.landmarks.size() << " landmarks" << endl;
                for (auto landmark = pose.landmarks.begin(); landmark != pose.landmarks.end(); ++landmark) {
                    cout << "      xyz = " << landmark->x << ", " << landmark->y << ", " << landmark->z<< endl;
                }
            }
            cout << "  world poses           : " << result->pose_world_landmarks.size() << endl;
        }
      };

  auto landmarker = PoseLandmarker::Create(std::move(cpp_options));
  if (!landmarker.ok()) {
    cerr << "Failed to create PoseLandmarker: " << landmarker.status() << endl;
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
