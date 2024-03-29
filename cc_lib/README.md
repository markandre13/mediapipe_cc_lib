# Mediapipe Standalone C++ Library

### introduction

while mediapipe is written in c++, it does not build a c++ library. (but there are an
internal c api, and external python, java, javascript and objc api's.)

[mediapipe_cpp_lib](https://github.com/purgeme/mediapipe_cpp_lib) solved that but
on 2024-02-19 was put on pause due to a new api in mediapipe using 'tasks'.

therefore i had to come up with something on my own.

* only macos (too early to worry about cross-platform)
* only face_landmarker (that's what i needed 1st)
* api is close to mediapipe's internal api's
* has it's own headers and mediapipe's internal one contain
  to many dependencies

### build

from the root directory run

    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe

one can also build a shared library but it's an experimental bazel features and takes long to run

    bazel build --experimental_cc_shared_library -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_shared

### develop

to try things out, put a main() into `cc_lib/mediapipe.cc` and run

    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 cc_lib:mediapipe_bin && ./bazel-mediapipe_cc_lib/bazel-out/darwin-opt/bin/cc_lib/mediapipe_bin
