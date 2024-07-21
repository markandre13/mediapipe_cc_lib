#include "mediapipe.hh"

// cv::imread
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"

// cv::VideoCapture
#include "mediapipe/framework/port/opencv_video_inc.h"

// https://docs.opencv.org/4.x/d7/dfc/group__highgui.html
#include "mediapipe/framework/port/opencv_highgui_inc.h"

#include <iostream>

using mediapipe::cc_lib::vision::core::RunningMode;
using mediapipe::cc_lib::vision::pose_landmarker::PoseLandmarker;
using mediapipe::cc_lib::vision::pose_landmarker::PoseLandmarkerOptions;

using namespace std;

int main() {
  cout << "start pose_landmarker_it" << endl;

  auto cpp_options = std::make_unique<PoseLandmarkerOptions>();
  cpp_options->base_options.model_asset_path =
      "cc_lib/pose_landmarker_heavy.task";
  cpp_options->running_mode = RunningMode::IMAGE;
  cpp_options->num_poses = 1;
  cpp_options->min_pose_detection_confidence = 0.5;
  cpp_options->min_pose_presence_confidence = 0.5;
  cpp_options->min_tracking_confidence = 0.5;
  cpp_options->output_segmentation_masks = false;

  auto landmarker = PoseLandmarker::Create(std::move(cpp_options));
  if (!landmarker) {
    cerr << "Failed to create PoseLandmarker" << endl;
    return 1;
  }

  // https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8
  cv::Mat input =
      cv::imread("mediapipe/model_maker/python/vision/object_detector/testdata/"
                 "coco_data/images/000000000431.jpg");
  if (input.data == nullptr) {
    cerr << "failed to load image" << endl;
    return 1;
  }

  auto result = landmarker->Detect(input.channels(), input.cols, input.rows,
                                   input.step, input.data);
  if (!result.has_value()) {
    cerr << "failed to detect pose landmarks" << endl;
    return 1;
  }

  //
  // pose landmarks
  //

  if (result->pose_landmarks.size() != 1) {
    cerr << "face landmarks: expected one pose" << endl;
    return 1;
  }

  if (result->pose_landmarks[0].landmarks.size() != 33) {
    cerr << "pose landmarks: expected 33 landmarks for pose, found "
         << result->pose_landmarks[0].landmarks.size() << endl;
    return 1;
  }
  auto &lm = result->pose_landmarks[0].landmarks[0];
  if (fabs(lm.x - 0.358083) > 0.000001) {
    cerr << "face landmarks: expected x = 0.358083, got " << lm.x << endl;
    return 1;
  }
  if (fabs(lm.y - 0.249386) > 0.000001) {
    cerr << "face landmarks: expected y = 0.249386, got " << lm.y << endl;
    return 1;
  }
  if (fabs(lm.z - -0.305153) > 0.000001) {
    cerr << "face landmarks: expected z = -0.305153, got " << lm.z << endl;
    return 1;
  }

  if (result->pose_world_landmarks.size() != 1) {
    cerr << "face landmarks: expected one world pose" << endl;
    return 1;
  }

  if (result->pose_world_landmarks[0].landmarks.size() != 33) {
    cerr << "world pose landmarks: expected 33 landmarks for pose, found "
         << result->pose_world_landmarks[0].landmarks.size() << endl;
    return 1;
  }
  lm = result->pose_world_landmarks[0].landmarks[0];
  if (fabs(lm.x - 0.134793) > 0.000001) {
    cerr << "world face landmarks: expected x = 0.134793, got " << lm.x << endl;
    return 1;
  }
  if (fabs(lm.y - -0.485992) > 0.000001) {
    cerr << "world face landmarks: expected y = -0.485992, got " << lm.y << endl;
    return 1;
  }
  if (fabs(lm.z - -0.3533) > 0.000001) {
    cerr << "world face landmarks: expected z = -0.3533, got " << lm.z << endl;
    return 1;
  }

  cout << "finish pose_landmarker_it" << endl;

  return 0;
}
