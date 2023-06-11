NVCC        = nvcc
OPENCV_INC  = -I/usr/include/opencv    # Specify the OpenCV include path
OPENCV_LIBS = -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab   # Specify the necessary OpenCV libraries

ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 
else
NVCC_FLAGS  = -O3 --std=c++03
endif
LD_FLAGS    = -lcudart $(OPENCV_LIBS)
EXE         = jpeg
OBJ         = main.o

default: $(EXE)

main.o: ./src/main.cu ./src/kernel.cu
	$(NVCC) -c -o $@ ./src/main.cu $(NVCC_FLAGS) $(OPENCV_INC)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)