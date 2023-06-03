
NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 
else
NVCC_FLAGS  = -O3 --std=c++03
endif
LD_FLAGS    = -lcudart
EXE	        = jpeg
OBJ	        = main.o 

default: $(EXE)

main.o: main.cu kernel.cu 
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

# support.o: support.cu support.h
# 	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
