#include "mediapipe_detail.hh"

using namespace std;

namespace mediapipe {
namespace cc_lib {
namespace detail {

std::unique_ptr<mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerOptions> convert(
    const std::unique_ptr<mediapipe::cc_lib::vision::pose_landmarker::PoseLandmarkerOptions> &in) {
    auto out = std::make_unique<mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerOptions>();

    out->base_options.model_asset_path = in->base_options.model_asset_path;
    out->running_mode = static_cast<mediapipe::tasks::vision::core::RunningMode>(in->running_mode);

    out->num_poses = in->num_poses;
    out->min_pose_detection_confidence = in->min_pose_detection_confidence;
    out->min_pose_presence_confidence = in->min_pose_presence_confidence;
    out->min_tracking_confidence = in->min_tracking_confidence;
    out->output_segmentation_masks = in->output_segmentation_masks;

    if (in->result_callback != nullptr) {
        auto rb = in->result_callback;
        out->result_callback = [rb](absl::StatusOr<mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult> result, const Image &,
                                    int64_t timestamp_ms) {
            if (result.ok()) {
                rb(convert(*result), timestamp_ms);
            }
        };
    }
    return out;
}

::mediapipe::cc_lib::vision::pose_landmarker::PoseLandmarkerResult convert(const mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult &in) {
    ::mediapipe::cc_lib::vision::pose_landmarker::PoseLandmarkerResult out;

    {
        out.pose_landmarks.resize(in.pose_landmarks.size());
        auto poseOut = out.pose_landmarks.begin();
        for (auto poseIn = in.pose_landmarks.begin(); poseIn != in.pose_landmarks.end(); ++poseIn, ++poseOut) {
            poseOut->landmarks.resize(poseIn->landmarks.size());
            auto landmarkOut = poseOut->landmarks.begin();
            for (auto landmarkIn = poseIn->landmarks.begin(); landmarkIn != poseIn->landmarks.end(); ++landmarkIn, ++landmarkOut) {
                landmarkOut->x = landmarkIn->x;
                landmarkOut->y = landmarkIn->y;
                landmarkOut->z = landmarkIn->z;
            }
        }
    }

    {
        out.pose_world_landmarks.resize(in.pose_world_landmarks.size());
        auto poseOut = out.pose_world_landmarks.begin();
        for (auto poseIn = in.pose_world_landmarks.begin(); poseIn != in.pose_world_landmarks.end(); ++poseIn, ++poseOut) {
            poseOut->landmarks.resize(poseIn->landmarks.size());
            auto landmarkOut = poseOut->landmarks.begin();
            for (auto landmarkIn = poseIn->landmarks.begin(); landmarkIn != poseIn->landmarks.end(); ++landmarkIn, ++landmarkOut) {
                landmarkOut->x = landmarkIn->x;
                landmarkOut->y = landmarkIn->y;
                landmarkOut->z = landmarkIn->z;
            }
        }
    }

    return out;
}

}  // namespace detail

namespace vision {
namespace pose_landmarker {

PoseLandmarkerOptions::PoseLandmarkerOptions() {}

std::unique_ptr<PoseLandmarker> PoseLandmarker::Create(std::unique_ptr<PoseLandmarkerOptions> options) {
    auto plm = ::mediapipe::tasks::vision::pose_landmarker::PoseLandmarker::Create(std::move(::mediapipe::cc_lib::detail::convert(options)));
    if (!plm.ok()) {
        return {};
    }
    auto result = std::make_unique<PoseLandmarker>();
    result->mp = std::move(*plm);
    return result;
}

PoseLandmarker::~PoseLandmarker() {}

std::optional<PoseLandmarkerResult> PoseLandmarker::Detect(int channels, int width, int height, int width_step, uint8_t *pixel_data) {
    ImageFrame image_frame(channels == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB, width, height, width_step, pixel_data, [](uint8_t *) {});
    Image image(std::make_shared<mediapipe::ImageFrame>(std::move(image_frame)));

    auto mp_result = mp->Detect(image);

    if (!mp_result.ok()) {
        cerr << "Detection failed: " << mp_result.status() << endl;
        return std::nullopt;
    }

    return ::mediapipe::cc_lib::detail::convert(*mp_result);
}

std::optional<PoseLandmarkerResult> PoseLandmarker::DetectForVideo(int channels, int width, int height, int width_step, uint8_t *pixel_data,
                                                                   int64_t timestamp_ms) {
    ImageFrame image_frame(channels == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB, width, height, width_step, pixel_data, [](uint8_t *) {});
    Image image(std::make_shared<mediapipe::ImageFrame>(std::move(image_frame)));

    auto mp_result = mp->DetectForVideo(image, timestamp_ms);

    if (!mp_result.ok()) {
        cerr << "Detection failed: " << mp_result.status() << endl;
        return std::nullopt;
    }

    return ::mediapipe::cc_lib::detail::convert(*mp_result);
}

bool PoseLandmarker::DetectAsync(int channels, int width, int height, int width_step, uint8_t *pixel_data, int64_t timestamp_ms) {
    ImageFrame image_frame(channels == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB, width, height, width_step, pixel_data, [](uint8_t *) {});
    Image image(std::make_shared<mediapipe::ImageFrame>(std::move(image_frame)));

    auto mp_result = mp->DetectAsync(image, timestamp_ms);

    if (!mp_result.ok()) {
        cerr << "Detection failed: " << mp_result << endl;
        return false;
    }

    return true;
}

bool PoseLandmarker::Close() {
    auto mp_result = mp->Close();
    return mp_result.ok();
}

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace cc_lib
}  // namespace mediapipe
