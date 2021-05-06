# LU Makefile

CXX=g++
CXXFLAGS=-I../common
NVCC=nvcc
NVCCFLAGS=-I../common

.PHONY: all
.SUFFIXES: .cpp .cu .cuh .o .seq .cbin .omp

all: seq omp cuda

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $<
.cu.o:
	$(NVCC) -c $(CXXFLAGS) $<


# Binary Production Rules
.cpp.seq: $(OBJS) 
	$(CXX) $< $(OBJS) $(CXXFLAGS) -o $@

.cpp.omp: $(OBJS)
	$(CXX) $< $(OBJS) $(CXXFLAGS) -fopenmp -o $@
# Doesn't do standalone compilation of entry point object, but OK

.cpp.cbin: $(OBJS)
	$(CXX) $< $(OBJS) $(NVCCFLAGS) -o $@

seq: lu_easy.seq 

omp: lu_par.omp

cuda: lu_cuda.cbin
