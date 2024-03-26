# Mediapipe Standalone C++ Library

in the root directory run

    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe
    bazel build --experimental_cc_shared_library -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_shared

to create `bazel-bin/cc_lib/libmediapipe.dylib`
