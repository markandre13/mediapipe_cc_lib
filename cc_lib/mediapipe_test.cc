#include "mediapipe/framework/port/gtest.h"
#include "mediapipe_detail.hh"

TEST(MediapipeCcLib, RunningMode) {
    ASSERT_EQ(mediapipe::cc_lib::vision::core::RunningMode::IMAGE, mediapipe::tasks::vision::core::RunningMode::IMAGE);
    ASSERT_EQ(mediapipe::cc_lib::vision::core::RunningMode::VIDEO, mediapipe::tasks::vision::core::RunningMode::VIDEO);
    ASSERT_EQ(mediapipe::cc_lib::vision::core::RunningMode::LIVE_STREAM, mediapipe::tasks::vision::core::RunningMode::LIVE_STREAM);
}

TEST(MediapipeCcLib, FaceLandmarkerOptions_SameDefaults) {
    auto mp_options = std::make_unique<mediapipe::tasks::vision::face_landmarker::FaceLandmarkerOptions>();
    auto options = std::make_unique<mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerOptions>();
   
    ASSERT_EQ(options->base_options.model_asset_path, mp_options->base_options.model_asset_path);
    ASSERT_EQ(options->running_mode, mp_options->running_mode);
    ASSERT_EQ(options->num_faces, mp_options->num_faces);
    ASSERT_EQ(options->min_face_detection_confidence, mp_options->min_face_detection_confidence);
    ASSERT_EQ(options->min_face_presence_confidence, mp_options->min_face_presence_confidence);
    ASSERT_EQ(options->min_tracking_confidence, mp_options->min_tracking_confidence);
    ASSERT_EQ(options->output_face_blendshapes, mp_options->output_face_blendshapes);
    ASSERT_EQ(options->output_facial_transformation_matrixes, mp_options->output_facial_transformation_matrixes);
}

TEST(MediapipeCcLib, FaceLandmarkerOptions_SameValues) {
    auto options = std::make_unique<mediapipe::cc_lib::vision::face_landmarker::FaceLandmarkerOptions>();
    options->base_options.model_asset_path += "face_landmarker_v2_with_blendshapes.task";

    ASSERT_EQ(options->running_mode, mediapipe::cc_lib::vision::core::RunningMode::IMAGE);
    options->running_mode = mediapipe::cc_lib::vision::core::RunningMode::LIVE_STREAM;

    options->num_faces += 1;
    options->min_face_detection_confidence += 0.1;
    options->min_face_presence_confidence += 0.2;
    options->min_tracking_confidence = 0.3;
    options->output_face_blendshapes = !options->output_face_blendshapes;
    options->output_facial_transformation_matrixes = !options->output_facial_transformation_matrixes;

    auto mp_options = mediapipe::cc_lib::detail::convert(options);

    ASSERT_EQ(options->base_options.model_asset_path, mp_options->base_options.model_asset_path);
    ASSERT_EQ(options->running_mode, mp_options->running_mode);
    ASSERT_EQ(options->num_faces, mp_options->num_faces);
    EXPECT_FLOAT_EQ(options->min_face_detection_confidence, mp_options->min_face_detection_confidence);
    EXPECT_FLOAT_EQ(options->min_face_presence_confidence, mp_options->min_face_presence_confidence);
    EXPECT_FLOAT_EQ(options->min_tracking_confidence, mp_options->min_tracking_confidence);
    ASSERT_EQ(options->output_face_blendshapes, mp_options->output_face_blendshapes);
    ASSERT_EQ(options->output_facial_transformation_matrixes, mp_options->output_facial_transformation_matrixes);
}

TEST(MediapipeCcLib, FaceLandmarkerResult_SameValues) {
    mediapipe::tasks::vision::face_landmarker::FaceLandmarkerResult in {
        .face_landmarks {
            {
                .landmarks {
                    {.x = 1.0, .y = 1.1, .z = 1.2 },
                    {.x = 2.0, .y = 2.1, .z = 2.2 },
                    {.x = 3.0, .y = 3.1, .z = 3.2 },    
                }
            }
        },
        .face_blendshapes {
            {{
                .categories { {
                    .index = 0,
                    .score = 0.6,
                    .category_name { "jawOpen" }
                }, {
                    .index = 1,
                    .score = 0.4,
                    .category_name { "mouthClosed" }
                } },
                .head_index = 0,
                .head_name { "first" }
            }}
        }
    };

    auto out = mediapipe::cc_lib::detail::convert(in);

    ASSERT_EQ(out.face_landmarks.size(), 1);
    ASSERT_EQ(out.face_landmarks[0].landmarks.size(), 3);
    EXPECT_FLOAT_EQ(out.face_landmarks[0].landmarks[0].x, 1.0);
    EXPECT_FLOAT_EQ(out.face_landmarks[0].landmarks[1].y, 2.1);
    EXPECT_FLOAT_EQ(out.face_landmarks[0].landmarks[2].z, 3.2);

    ASSERT_TRUE(out.face_blendshapes.has_value());
    ASSERT_EQ(out.face_blendshapes->size(), 1);
    ASSERT_EQ(out.face_blendshapes->at(0).categories.size(), 2);

    ASSERT_EQ(out.face_blendshapes->at(0).categories[0].index, 0);
    EXPECT_FLOAT_EQ(out.face_blendshapes->at(0).categories[0].score, 0.6);
    ASSERT_TRUE(out.face_blendshapes->at(0).categories[0].category_name.has_value());
    ASSERT_EQ(*out.face_blendshapes->at(0).categories[0].category_name, "jawOpen");

    ASSERT_EQ(out.face_blendshapes->at(0).categories[1].index, 1);
    EXPECT_FLOAT_EQ(out.face_blendshapes->at(0).categories[1].score, 0.4);
    ASSERT_TRUE(out.face_blendshapes->at(0).categories[1].category_name.has_value());
    ASSERT_EQ(*out.face_blendshapes->at(0).categories[1].category_name, "mouthClosed");
}
