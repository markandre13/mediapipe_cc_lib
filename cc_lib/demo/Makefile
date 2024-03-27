APP=demo
OUT_DIR=../../bazel-bin/cc_lib

CXX=/usr/local/opt/llvm/bin/clang++
CFLAGS=-std=c++23 \
        -Wall -Wextra -Werror=return-type -Werror=shadow \
        -Wno-deprecated-anon-enum-enum-conversion -Wno-module-import-in-extern-c -Wno-unused-parameter \
        -O0 -g -I/usr/local/opt/llvm/include/c++ \
		-I../.. \
		-I../../bazel-bin \
		-I../../bazel-mediapipe/external/com_google_absl \
		-I../../bazel-mediapipe/external/com_github_glog_glog/src \
		-I../../bazel-bin/external/com_github_glog_glog/src \
		-I../../bazel-bin/external/com_github_gflags_gflags/_virtual_includes/gflags \
		-I../../bazel-mediapipe/external/com_google_protobuf/src
LDFLAGS=-L$(OUT_DIR) -lmediapipe

SRC=demo.cc

OBJ = $(SRC:.cc=.o)

all: $(APP)

run:
	DYLD_LIBRARY_PATH=$(OUT_DIR) ./$(APP)

depend:
	makedepend -I$(CORBA_PATH) -Y $(SRC)

.SUFFIXES: .idl .cc .c .h .hh .o

#$(APP): demo.cc
# g++ -I../.. demo.cc -L$(OUT_DIR) -o $(APP) -lmediapipe

$(APP): $(OBJ)
	@echo "linking..."
	$(CXX) $(LDFLAGS) $(OBJ) -o $(APP)

.cc.o:
	@echo compiling $*.cc ...
	$(CXX) $(CFLAGS) -c -o $*.o $*.cc
