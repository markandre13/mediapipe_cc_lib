# Mediapipe Standalone C++ Library

this creates a standalone c++ library of mediapipe in  `bazel-bin/cc_lib/`

in the root directory run

    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe

one can also build a shared library but it's an experimental bazel features and takes long to run

    bazel build --experimental_cc_shared_library -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_shared

