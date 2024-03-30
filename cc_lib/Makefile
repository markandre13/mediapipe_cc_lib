all:
	cd .. && bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe

test:
	cd .. && bazel test --test_output=all --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_test
	
run:
	bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_main && ./bazel-mediapipe_cc_lib/bazel-out/darwin-opt/bin/cc_lib/mediapipe_main
