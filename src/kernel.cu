#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#define BLOCK_SIZE 8

using namespace cv;
using namespace std;

//forward transform
__constant__ float dctMatrix[BLOCK_SIZE * BLOCK_SIZE] = {
    0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
    0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904,
    0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619,
    0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157,
    0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536,
    0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778,
    0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913,
    0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975
};

//inverse transform
__constant__ float IDctMatrix[BLOCK_SIZE * BLOCK_SIZE] = {
    0.3536, 0.4904, 0.4619, 0.4157, 0.3536, 0.2778, 0.1913, 0.0975,
    0.3536, 0.4157, 0.1913, -0.0975, -0.3536, -0.4904, -0.4619, -0.2778,
    0.3536, 0.2778, -0.1913, -0.4904, -0.3536, 0.0975, 0.4619, 0.4157,
    0.3536, 0.0975, -0.4619, -0.2778, 0.3536, 0.4157, -0.1913, -0.4904,
    0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536,
    0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778,
    0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913,
    0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975
};

__global__ void DCT(int row, int col, const float *d_image) {

    extern __shared__ float cache_one[];
    extern __shared__ float cache_two[];   


    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (dim x dim) matrix
     *   where B is a (dim x dim) matrix
     *   where C is a (dim x dim) matrix
     *
     ********************************************************************/


    /*************************************************************************/
    // INSERT KERNEL CODE HERE

    /*************************************************************************/







}

void LaunchDCT(const int row, const int col, const float *d_image)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksPerGrid(ceil(row/(float)threadsPerBlock.x), ceil(col/(float)threadsPerBlock.y), 1);
    DCT<<<blocksPerGrid, BLOCK_SIZE, sizeof(float)*BLOCK_SIZE*BLOCK_SIZE*2>>>(row, col, d_image, result);

}

