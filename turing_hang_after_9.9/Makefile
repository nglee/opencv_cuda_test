CV_REL_INS=/usr/local/opencv/release
CV_REL_INC=$(CV_REL_INS)/include
CV_REL_LIB=$(CV_REL_INS)/lib

CU_INC=/usr/local/cuda/include
CU_LIB=/usr/local/cuda/lib64

REL_OPT=-I$(CV_REL_INC) -L$(CV_REL_LIB) -Wl,-rpath,$(CV_REL_LIB) -I$(CU_INC) -L$(CU_LIB)

all: template template2 integral warp haar

template: template.cpp
	g++ -o $@ $< $(REL_OPT) -lopencv_core -lopencv_cudaimgproc -lopencv_cudaarithm -lcudart

template2: template2.cpp
	g++ -o $@ $< $(REL_OPT) -lopencv_core -lopencv_cudaimgproc -lopencv_cudaarithm -lcudart

integral: integral.cpp
	g++ -o $@ $< $(REL_OPT) -lopencv_core -lopencv_cudaimgproc -lopencv_cudaarithm -lcudart

warp: warp.cu
	nvcc -o $@ warp.cu

haar: haar.cpp
	g++ -o $@ $< $(REL_OPT) -lopencv_core -lopencv_cudaobjdetect -lopencv_cudalegacy

clean:
	rm -f template template2 integral warp haar
