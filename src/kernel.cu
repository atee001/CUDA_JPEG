#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#define TILE_SIZE 8

__global__ void matAdd(int dim, const float *A, const float *B, float* C) {

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

void basicMatAdd(int dim, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocksPerGrid(ceil(dim/(float)threadsPerBlock.x), ceil(dim/(float)threadsPerBlock.y), 1);
    matAdd<<<blocksPerGrid, TILE_SIZE, sizeof(float)*TILE_SIZE*TILE_SIZE*2>>>(dim, A, B, C);

}

