#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#define TILE_SIZE 16

__global__ void matAdd(int dim, const float *A, const float *B, float* C) {

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

void basicMatAdd(int dim, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocksPerDim(ceil(dim/(float)threadsPerBlock.x), ceil(dim/(float)threadsPerBlock.y), 1);
    matAdd<<<blocksPerDim, TILE_SIZE>>>(dim, A, B, C);


}

