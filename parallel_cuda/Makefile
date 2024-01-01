NVCC=nvcc
CXX=g++
CXXFLAGS=-I./common -Iobjs/ -O3 -std=c++17 -Wall -g -fPIC -lm

APP_NAME=KMeans
OBJDIR=.
COMMONDIR=.

CUDA_LINK_FLAGS=-rdc=true -Xcompiler '-fPIC' 
CUDA_COMPILE_FLAGS=--device-c -Xcompiler '-fPIC' -g -O3
#-gencode=arch=compute_61,code=sm_61 

default: $(APP_NAME)

.PHONY: dirs clean

# dirs:
# 	/bin/mkdir -p $(OBJDIR)/

clean:
	rm *.o $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/kernel.o

$(APP_NAME): $(OBJS)
	$(NVCC) ${CUDA_LINK_FLAGS} -o $@ $(OBJS)

$(OBJDIR)/%.o: %.c
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: $(COMMONDIR)/%.c
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/kernel.o : kernel.cu kernel.h
	${NVCC} ${CUDA_COMPILE_FLAGS} -c kernel.cu -o $@

# SRC_DIR=..

# all: clean
# 	mkdir build && \
# 	cd build && \
# 	nvcc --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3 $(SRC_DIR)/kernel.cu && \
# 	gcc $(SRC_DIR)/main.c kernel.o -o KMeans

# clean:
# 	rm -r build
