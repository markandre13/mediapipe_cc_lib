#ifndef MEDIAPIPE_CCLIB_DETAIL_H_
#define MEDIAPIPE_CCLIB_DETAIL_H_

#include "mediapipe.hh"

#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/face_landmarker_result.h"

namespace mediapipe {
namespace cc_lib {
namespace detail {

std::unique_ptr<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>
convert(const std::unique_ptr<mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerOptions> &in);

mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerResult
convert(const mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult &);

}
} // namespace cc_lib
} // namespace mediapipe

#endif
