#ifndef MEDIAPIPE_CCLIB_H_
#define MEDIAPIPE_CCLIB_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>
// #include <expected>

namespace mediapipe {

namespace tasks {
namespace vision {
namespace face_landmarker {
class FaceLandmarker;
}  // namespace face_landmarker
}  // namespace vision
}  // namespace tasks

namespace cc_lib {
namespace core {

// mediapipe/tasks/cc/core/base_options.h
struct BaseOptions {
        // The path to the model asset to open and mmap in memory.
        std::string model_asset_path = "";
};

}  // namespace core

namespace components {
namespace containers {

// wrapper for mediapipe/tasks/cc/components/containers/landmark.h

/**
 * NormalizedLandmark represents a point in 3D space with x, y, z coordinates.
 * The coordinates are within [0, 1]. z represents the landmark depth, and the
 * smaller the value the closer the world landmark is to the camera.
 */
struct NormalizedLandmark {
        float x;
        float y;
        float z;

        // not set by face_landmarker_v2_with_blendshapes.task, hence skipped
        // std::optional<float> visibility = std::nullopt;
        // std::optional<float> presence = std::nullopt;
        // std::optional<std::string> name = std::nullopt;
};
struct NormalizedLandmarks {
        std::vector<NormalizedLandmark> landmarks;
};

}  // namespace containers
}  // namespace components

namespace vision {
namespace core {

/**
 *  The running mode of a MediaPipe vision task.
 */
enum RunningMode {
    /**
     * Run the vision task on a single image input.
     */
    IMAGE = 1,

    /**
     * Run the vision task on the decoded frames of an input video.
     */
    VIDEO = 2,

    /**
     * Run the vision task on a live stream of input data, such as from camera.
     */
    LIVE_STREAM = 3,
};

}  // namespace core

namespace face_landmarker {

// wrapper for mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h

/**
 * The face landmarks detection result from FaceLandmarker, where each vector
 * element represents a single face detected in the image.
 */
struct FaceLandmarkerResult {
        /**
         * Detected face landmarks in normalized image coordinates.
         */
        std::vector<::mediapipe::cc_lib::components::containers::NormalizedLandmarks> face_landmarks;
        //   // Optional face blendshapes results.
        //   std::optional<std::vector<components::containers::Classifications>>
        //       face_blendshapes;
        //   // Optional facial transformation matrix.
        //   std::optional<std::vector<Matrix>> facial_transformation_matrixes;
};

// wrapper for mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h:FaceLandmarkerOptions

/**
 * Configuration options for FaceLandmarker
 */
struct FaceLandmarkerOptions {
        FaceLandmarkerOptions();

        /**
         * Base options for configuring MediaPipe Tasks library, such as specifying
         * the TfLite model bundle file with metadata, accelerator options, op
         * resolver, etc.
         */
        ::mediapipe::cc_lib::core::BaseOptions base_options;

        /**
         * The running mode of the task. Default to the image mode.
         * FaceLandmarker has three running modes:
         * 1) The image mode for detecting face landmarks on single image inputs.
         * 2) The video mode for detecting face landmarks on the decoded frames of a
         *    video.
         * 3) The live stream mode for detecting face landmarks on the live stream of
         *    input data, such as from camera. In this mode, the "result_callback"
         *    below must be specified to receive the detection results asynchronously.
         */
        ::mediapipe::cc_lib::vision::core::RunningMode running_mode = mediapipe::cc_lib::vision::core::RunningMode::IMAGE;

        /**
         * The maximum number of faces that can be detected by the FaceLandmarker.
         */
        int num_faces = 1;

        /**
         * The minimum confidence score for the face detection to be considered successful.
         */
        float min_face_detection_confidence = 0.5;

        /**
         * The minimum confidence score of face presence score in the face landmark detection.
         */
        float min_face_presence_confidence = 0.5;

        /**
         * The minimum confidence score for the face tracking to be considered successful.
         */
        float min_tracking_confidence = 0.5;

        /**
         * Whether FaceLandmarker outputs face blendshapes classification.
         * Face blendshapes are used for rendering the 3D face model.
         */
        bool output_face_blendshapes = false;

        /**
         * Whether FaceLandmarker outputs facial transformation_matrix. Facial
         * transformation matrix is used to transform the face landmarks in canonical
         * face to the detected face, so that users can apply face effects on the
         * detected landmarks.
         */
        bool output_facial_transformation_matrixes = false;

        /**
         * The user-defined result callback for processing live stream data.
         * The result callback should only be specified when the running mode is set
         * to RunningMode::LIVE_STREAM.
         */
        std::function<void(std::optional<FaceLandmarkerResult>, int64_t timestamp_ms)> result_callback = nullptr;
};

class FaceLandmarker {
    public:
        ~FaceLandmarker();

        std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarker> mp;
        // static std::expected<std::unique_ptr<FaceLandmarker>, std::runtime_error> Create(std::unique_ptr<FaceLandmarkerOptions> options);
        static std::unique_ptr<FaceLandmarker> Create(std::unique_ptr<FaceLandmarkerOptions> options);

        std::optional<FaceLandmarkerResult> Detect(int channels, int width, int height, int width_step, uint8_t* pixel_data);
        std::optional<FaceLandmarkerResult> DetectForVideo(int channels, int width, int height, int width_step, uint8_t* pixel_data, int64_t timestamp_ms);
        bool DetectAsync(int channels, int width, int height, int width_step, uint8_t* pixel_data, int64_t timestamp_ms);
};

}  // namespace face_landmarker
}  // namespace vision
}  // namespace cc_lib
}  // namespace mediapipe

#endif
