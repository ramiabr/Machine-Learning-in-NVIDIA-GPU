# C compiler
CC = gcc
CC_FLAGS = -g -O2 

NVCC = nvcc 
NVCC_FLAGS =  -O3 -I/usr/local/cuda/include -arch=sm_20
NVCC_LD_FLAGS =  -L /usr/local/cuda/lib 

backprop: backprop.o facetrain.o imagenet.o backprop_kernel.o gpuwrapper.o support.o 
	$(NVCC) $(NVCC_LD_FLAGS) backprop.o facetrain.o imagenet.o backprop_kernel.o gpuwrapper.o support.o -o backprop -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_kernel.o: backprop_kernel.c backprop.h
	$(CC) $(CC_FLAGS) backprop_kernel.c -c

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c

gpuwrapper.o: gpuwrapper.cu 
	$(NVCC) $(NVCC_FLAGS)  -c -o $@ gpuwrapper.cu

support.o: support.cu 
	$(NVCC) $(NVCC_FLAGS)  -c -o $@ support.cu

clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
