# build library
all: face_landmarker.task pose_landmarker_lite.task pose_landmarker_full.task pose_landmarker_heavy.task
	cd .. && bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe

face_landmarker.task:
	wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

pose_landmarker_lite.task:
	wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

pose_landmarker_full.task:
	wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

pose_landmarker_heavy.task:
	wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

# unit tests
test:
	cd .. && bazel test --test_output=all --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_test

# integration test
run:
	cd .. && bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_main && ./bazel-mediapipe_cc_lib/bazel-out/darwin-opt/bin/cc_lib/mediapipe_main
