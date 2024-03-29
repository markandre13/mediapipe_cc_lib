#ifndef MEDIAPIPE_CCLIB_H_
#define MEDIAPIPE_CCLIB_H_

#include <string>

namespace mediapipe {
namespace cc_lib {
namespace core {

// mediapipe/tasks/cc/core/base_options.h
struct BaseOptions {
  // The path to the model asset to open and mmap in memory.
  std::string model_asset_path = "";
};

} // namespace core

namespace vision {

namespace core {

// The running mode of a MediaPipe vision task.
enum RunningMode {
  // Run the vision task on single image inputs.
  IMAGE = 1,

  // Run the vision task on the decoded frames of an input video.
  VIDEO = 2,

  // Run the vision task on a live stream of input data, such as from camera.
  LIVE_STREAM = 3,
};

} // namespace core

namespace face_landmarker {

struct FaceLandmarkerOptions {
    FaceLandmarkerOptions();

    // Base options for configuring MediaPipe Tasks library, such as specifying
    // the TfLite model bundle file with metadata, accelerator options, op
    // resolver, etc.
    ::mediapipe::cc_lib::core::BaseOptions base_options;

    // The running mode of the task. Default to the image mode.
    // FaceLandmarker has three running modes:
    // 1) The image mode for detecting face landmarks on single image inputs.
    // 2) The video mode for detecting face landmarks on the decoded frames of a
    //    video.
    // 3) The live stream mode for detecting face landmarks on the live stream of
    //    input data, such as from camera. In this mode, the "result_callback"
    //    below must be specified to receive the detection results asynchronously.
    ::mediapipe::cc_lib::vision::core::RunningMode running_mode = mediapipe::cc_lib::vision::core::RunningMode::IMAGE;

    // The maximum number of faces that can be detected by the FaceLandmarker.
    int num_faces = 1;

    // The minimum confidence score for the face detection to be considered
    // successful.
    float min_face_detection_confidence = 0.5;

    // The minimum confidence score of face presence score in the face landmark
    // detection.
    float min_face_presence_confidence = 0.5;

    // The minimum confidence score for the face tracking to be considered
    // successful.
    float min_tracking_confidence = 0.5;

    // Whether FaceLandmarker outputs face blendshapes classification. Face
    // blendshapes are used for rendering the 3D face model.
    bool output_face_blendshapes = false;

    // Whether FaceLandmarker outputs facial transformation_matrix. Facial
    // transformation matrix is used to transform the face landmarks in canonical
    // face to the detected face, so that users can apply face effects on the
    // detected landmarks.
    bool output_facial_transformation_matrixes = false;

    // The user-defined result callback for processing live stream data.
    // The result callback should only be specified when the running mode is set
    // to RunningMode::LIVE_STREAM.
    //   std::function<void(absl::StatusOr<FaceLandmarkerResult>, const Image &,
    //                      int64_t)>
    //       result_callback = nullptr;
};
} // namespace face_landmarker
} // namespace vision
} // namespace cc_lib
} // namespace mediapipe

#endif
