# LU Makefile

CXX=g++
CXXFLAGS=-I../common -DENABLE_MOCK
NVCC=nvcc
NVCCFLAGS=-I../common

.PHONY: all
.SUFFIXES: .cpp .cu .cuh .o .seq .cbin .omp

all: seq omp cuda

debug_all:
	$(MAKE) X="-g -DVERBOSE_DEBUG"

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(X) $<
.cu.o:
	$(NVCC) -c $(CXXFLAGS) $(X) $<


# Binary Production Rules
.cpp.seq: $(OBJS)
	$(CXX) $< $(OBJS) $(X) $(CXXFLAGS) -o $@

.cpp.omp: $(OBJS)
	$(CXX) $< $(OBJS) $(X) $(CXXFLAGS) -fopenmp -o $@
# Doesn't do standalone compilation of entry point object, but OK

.cpp.cbin: $(OBJS)
	$(CXX) $< $(OBJS) $(X) $(NVCCFLAGS) -o $@

seq: lu_easy.seq 

omp: lu_par.omp

cuda: lu_cuda.cbin

clean:
	rm *.seq || rm *.cbin || rm *.omp || true
