#include <stdio.h>

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
    //
    int i = blockIdx.x*blockDim.x +threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if((i < dim) && (j < dim)){        
        C[i][j] = A[i][j] + B[i][j];
    }
        
    /*************************************************************************/

}

void basicMatAdd(int dim, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    dim3 numBlocks((dim/threadsPerBlock.x)+1, (dim/threadsPerBlock.y) + 1, 1);  //can't have 0 blocks if there are 100 elements 100/256 = 0 
    
    /*************************************************************************/
	matAdd<<<numBlocks, threadsPerBlock>>>(dim, A, B, C);
	// Invoke CUDA kernel -----------------------------------------------------
    /*************************************************************************/
    //INSERT CODE HERE
	
    /*************************************************************************/

}

