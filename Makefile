NVCC        = nvcc
OPENCV_INC = -I/usr/include/opencv
OPENCV_LIBS = -I/user/include/opencv/lib

ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 
else
NVCC_FLAGS  = -O3  --std=c++03 
endif
LD_FLAGS    = -lcudart $(OPENCV_LIBS)
EXE	        = jpeg
OBJ	        = main.o 

default: $(EXE)

main.o: ./src/main.cu ./src/kernel.cu 
	$(NVCC) -c -o $@ ./src/main.cu $(NVCC_FLAGS) $(OPENCV_INC)

#support.o: support.cu support.h
#	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)