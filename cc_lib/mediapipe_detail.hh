#ifndef MEDIAPIPE_CCLIB_DETAIL_H_
#define MEDIAPIPE_CCLIB_DETAIL_H_

#include "mediapipe.hh"

#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"

namespace mediapipe {
namespace cc_lib {
namespace detail {

std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>
convert(const std::unique_ptr<mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerOptions> &in);

mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerResult
convert(const mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult &);

std::unique_ptr<mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerOptions>
convert(const std::unique_ptr<mediapipe::cc_lib::vision::pose_landmarker::PoseLandmarkerOptions> &in);

mediapipe::cc_lib::vision::pose_landmarker::PoseLandmarkerResult
convert(const mediapipe::tasks::vision::pose_landmarker::PoseLandmarkerResult &);

}
} // namespace cc_lib
} // namespace mediapipe

#endif
