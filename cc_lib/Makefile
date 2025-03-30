# build the library

# on macOS
# * brew install llvm opencv python@3.12 numpy
# * get bazel 6.5.0
#   https://github.com/bazelbuild/bazel/releases/tag/6.5.0
BAZEL_BIN=bazel-6.5.0
BAZEL_DEF=--define MEDIAPIPE_DISABLE_GPU=1

all: face_landmarker.task pose_landmarker_lite.task pose_landmarker_full.task pose_landmarker_heavy.task
	cd .. && HERMETIC_PYTHON_VERSION=3.12 \
		$(BAZEL_BIN) build \
		--verbose_failures \
		--sandbox_debug \
		-c opt \
		$(BAZEL_DEF) \
		cc_lib:mediapipe

MODEL_URL=https://storage.googleapis.com/mediapipe-models

face_landmarker.task:
	curl --output $@ $(MODEL_URL)/face_landmarker/face_landmarker/float16/1/$@

pose_landmarker_lite.task:
	curl --output $@ $(MODEL_URL)/pose_landmarker/pose_landmarker_lite/float16/latest/$@

pose_landmarker_heavy.task:
	curl --output $@ $(MODEL_URL)/pose_landmarker/pose_landmarker_heavy/float16/latest/$@

pose_landmarker_full.task:
	curl --output $@ $(MODEL_URL)/pose_landmarker/pose_landmarker_full/float16/latest/$@

# unit tests
test:
	cd .. && HERMETIC_PYTHON_VERSION=3.12 $(BAZEL_BIN) test --test_output=all $(BAZEL_DEF) cc_lib:mediapipe_test

# integration test
integration:
	cd .. && HERMETIC_PYTHON_VERSION=3.12 $(BAZEL_BIN) build -c opt $(BAZEL_DEF) cc_lib:mediapipe_main && ./bazel-mediapipe_cc_lib/bazel-out/darwin-opt/bin/cc_lib/mediapipe_main
