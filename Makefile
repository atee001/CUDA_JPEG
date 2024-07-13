NVCC        = nvcc
OPENCV_INC  = -I/usr/include/opencv4    # Specify the OpenCV include path
OPENCV_LIBS = -lopencv_imgcodecs -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab   # Specify the necessary OpenCV libraries

CXXFLAGS += -std=c++11

ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -std=c++11
else
NVCC_FLAGS  = -O3 -std=c++11
endif

LD_FLAGS    = -lcudart $(OPENCV_LIBS)
EXE         = jpeg
OBJ         = main.o

default: $(EXE)

main.o: ./src/main.cu ./src/kernel.cu
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS) $(OPENCV_INC)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(OPENCV_LIBS)

clean:
	rm -rf *.o $(EXE)
