CXX = g++
CC  = gcc
LD  = g++
F90  = ifort

.SUFFIXES: .o .cpp .ptx .cu

CUDA_TK  = /usr/local/cuda

OFLAGS = -O3 -g -Wall 
CXXFLAGS =   $(OFLAGS)  -Wstrict-aliasing=2 


NVCC = $(CUDA_TK)/bin/nvcc  
NVCCFLAGS  =  -Xptxas=-v  -arch=sm_35 -rdc=true 
NVCCLDFLAGS = -lcudadevrt -arch=sm_35 

INCLUDEPATH = ./include
CXXFLAGS  += -I$(INCLUDEPATH)
NVCCFLAGS += -I$(INCLUDEPATH)

CUDAKERNELSPATH = ./
CUDAKERNELS = treecode.o \
							buildTree.o \
							computeMultipoles.o \
							computeForces.o 

SRCPATH = ./
SRC = anyoption.cpp



OBJ = $(SRC:%.cpp=%.o)

PROG = treecode

AR = ar ruv
RANLIB = ranlib
RM = rm


all:	  $(OBJ)  $(CUDAPTX) $(PROG)
kernels:  $(CUDAPTX)

$(PROG): $(OBJ) $(CUDAKERNELS)
	$(NVCC)  $^ -o $@  $(NVCCLDFLAGS)

%.o: $(SRCPATH)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.ptx: $(CUDAKERNELSPATH)/%.cu
	$(NVCC) $(NVCCFLAGS) -ptx $< -o $@

%.o:  $(CUDAKERNELSPATH)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


clean:
	/bin/rm -rf *.o  $(PROG) *.a *.cu_o 

clean_all:
	/bin/rm -rf *.o  $(PROG) *.a *.cu_o  *~

%.o: Treecode.h







