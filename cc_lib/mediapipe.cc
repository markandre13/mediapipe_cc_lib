#include "mediapipe_detail.hh"

using namespace std;

namespace mediapipe {
namespace cc_lib {
namespace detail {

std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions> convert(
    const std::unique_ptr<mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerOptions> &in) {
    auto out = std::make_unique<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>();

    out->base_options.model_asset_path = in->base_options.model_asset_path;
    out->running_mode = static_cast<mediapipe::tasks::vision::core::RunningMode>(in->running_mode);
    out->num_faces = in->num_faces;
    out->min_face_detection_confidence = in->min_face_detection_confidence;
    out->min_face_presence_confidence = in->min_face_presence_confidence;
    out->min_tracking_confidence = in->min_tracking_confidence;
    out->output_face_blendshapes = in->output_face_blendshapes;
    out->output_facial_transformation_matrixes = in->output_facial_transformation_matrixes;
    if (in->result_callback != nullptr) {
        auto rb = in->result_callback;
        out->result_callback = [rb](absl::StatusOr<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult> result, const Image &,
                                    int64_t timestamp_ms) {
            if (result.ok()) {
                rb(convert(*result), timestamp_ms);
            }
        };
    }
    return out;
}

::mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerResult convert(const mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult &in) {
    // for now, to keep the implementation simple and avoid mediapipe internal details, we just copy all the data.
    // ideally, we might just provide access to the protobuffer structures which
    // mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.cc: ConvertToFaceLandmarkerResult()
    // uses to create ::mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult
    ::mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerResult out;

    {
        out.face_landmarks.resize(in.face_landmarks.size());
        auto faceOut = out.face_landmarks.begin();
        for (auto faceIn = in.face_landmarks.begin(); faceIn != in.face_landmarks.end(); ++faceIn, ++faceOut) {
            faceOut->landmarks.resize(faceIn->landmarks.size());
            auto landmarkOut = faceOut->landmarks.begin();
            for (auto landmarkIn = faceIn->landmarks.begin(); landmarkIn != faceIn->landmarks.end(); ++landmarkIn, ++landmarkOut) {
                landmarkOut->x = landmarkIn->x;
                landmarkOut->y = landmarkIn->y;
                landmarkOut->z = landmarkIn->z;
                // landmarkOut->visibility = landmarkIn->visibility;
                // landmarkOut->presence = landmarkIn->presence;
                // landmarkOut->name = landmarkIn->name;
            }
        }
    }

    if (in.face_blendshapes.has_value()) {
        out.face_blendshapes = {{}};
        out.face_blendshapes->resize(in.face_blendshapes->size());
        auto faceOut = out.face_blendshapes->begin();
        for (auto faceIn = in.face_blendshapes->begin(); faceIn != in.face_blendshapes->end(); ++faceIn, ++faceOut) {
            faceOut->head_name = faceIn->head_name;
            faceOut->head_index = faceIn->head_index;
            faceOut->categories.resize(faceIn->categories.size());
            auto catOut = faceOut->categories.begin();
            for(auto catIn = faceIn->categories.begin(); catIn != faceIn->categories.end(); ++catIn, ++catOut) {
                catOut->index = catIn->index;
                catOut->score = catIn->score;
                catOut->category_name = catIn->category_name;
                catOut->display_name = catIn->display_name;
            }
        }
    }
    
    // if (in.facial_transformation_matrixes.has_value()) {
    //     cout << "we have facial_transformation_matrixes" << endl;
    // }

    //   if (facial_transformation_matrixes_proto.has_value()) {
    //     result.facial_transformation_matrixes =
    //         std::vector<Matrix>(facial_transformation_matrixes_proto->size());
    //     std::transform(facial_transformation_matrixes_proto->begin(),
    //                    facial_transformation_matrixes_proto->end(),
    //                    result.facial_transformation_matrixes->begin(),
    //                    [](const mediapipe::MatrixData& matrix_proto) {
    //                      mediapipe::Matrix matrix;
    //                      MatrixFromMatrixDataProto(matrix_proto, &matrix);
    //                      return matrix;
    //                    });
    //   }
    return out;
}

}  // namespace detail

namespace vision {
namespace face_landmarker {

FaceLandmarkerOptions::FaceLandmarkerOptions() {}

std::unique_ptr<FaceLandmarker> FaceLandmarker::Create(std::unique_ptr<FaceLandmarkerOptions> options) {
    auto flm = ::mediapipe::tasks::vision::face_landmarker::FaceLandmarker::Create(std::move(::mediapipe::cc_lib::detail::convert(options)));
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

std::optional<FaceLandmarkerResult> FaceLandmarker::Detect(int channels, int width, int height, int width_step, uint8_t *pixel_data) {
    ImageFrame image_frame(channels == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB, width, height, width_step, pixel_data, [](uint8_t *) {});
    Image image(std::make_shared<mediapipe::ImageFrame>(std::move(image_frame)));

    auto mp_result = mp->Detect(image);

    if (!mp_result.ok()) {
        cerr << "Detection failed: " << mp_result.status() << endl;
        return std::nullopt;
    }

    return ::mediapipe::cc_lib::detail::convert(*mp_result);
}

std::optional<FaceLandmarkerResult> FaceLandmarker::DetectForVideo(int channels, int width, int height, int width_step, uint8_t *pixel_data,
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

bool FaceLandmarker::DetectAsync(int channels, int width, int height, int width_step, uint8_t *pixel_data, int64_t timestamp_ms) {
    ImageFrame image_frame(channels == 4 ? ImageFormat::SRGBA : ImageFormat::SRGB, width, height, width_step, pixel_data, [](uint8_t *) {});
    Image image(std::make_shared<mediapipe::ImageFrame>(std::move(image_frame)));

    auto mp_result = mp->DetectAsync(image, timestamp_ms);

    if (!mp_result.ok()) {
        cerr << "Detection failed: " << mp_result << endl;
        return false;
    }

    return true;
}

}  // namespace face_landmarker
}  // namespace vision
}  // namespace cc_lib
}  // namespace mediapipe
