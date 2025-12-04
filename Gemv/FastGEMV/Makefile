EXECUTABLE := gemv

CU_FILES   := test_gemv.cu fast_gemv.cu utility.cu

CU_DEPS    := fast_gemv.cuh utility.cuh

CC_FILES   := main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -std=c++17
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart -lcuda -lcurand
NVCC=nvcc
NVCCFLAGS=-O3 -std=c++17 -gencode arch=compute_80,code=sm_80

.PHONY: clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

clean:
	rm -rf $(EXECUTABLE)

$(EXECUTABLE): main.cpp test_gemv.cu fast_gemv.cu utility.cu $(CU_DEPS)
	$(NVCC) $(NVCCFLAGS) -o $@ main.cpp test_gemv.cu fast_gemv.cu utility.cu -lcudart -lcuda -lcurand