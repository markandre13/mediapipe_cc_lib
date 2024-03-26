APP=demo
OUT_DIR=../../bazel-bin/cc_lib

$(APP): demo.cc
	g++ demo.cc -L$(OUT_DIR) -o $(APP) -lmediapipe

run:
	DYLD_LIBRARY_PATH=$(OUT_DIR) ./$(APP)
