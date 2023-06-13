#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#define BLOCK_SIZE 8

using namespace cv;
using namespace std;

//DCT matrix T obtained from matlab dctmtx(8)
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

//transposed DCT matrix T' obtained from matlab dctmtx(8) with a transpose
__constant__ float IdctMatrix[BLOCK_SIZE * BLOCK_SIZE] = {
    0.3536, 0.4904, 0.4619, 0.4157, 0.3536, 0.2778, 0.1913, 0.0975,
    0.3536, 0.4157, 0.1913, -0.0975, -0.3536, -0.4904, -0.4619, -0.2778,
    0.3536, 0.2778, -0.1913, -0.4904, -0.3536, 0.0975, 0.4619, 0.4157,
    0.3536, 0.0975, -0.4619, -0.2778, 0.3536, 0.4157, -0.1913, -0.4904,
    0.3536, -0.0975, -0.4619, 0.2778, 0.3536, -0.4157, -0.1913, 0.4904,
    0.3536, -0.2778, -0.1913, 0.4904, -0.3536, -0.0975, 0.4619, -0.4157,
    0.3536, -0.4157, 0.1913, 0.0975, -0.3536, 0.4904, -0.4619, 0.2778,
    0.3536, -0.4904, 0.4619, -0.4157, 0.3536, -0.2778, 0.1913, -0.0975
};

//F(p,q) = T * f(x,y) * T'

__global__ void DCT(int numRows, int numCols, float *d_image, float *f_image) {

    __shared__ float cache[BLOCK_SIZE*BLOCK_SIZE];  
    int y = threadIdx.y + (blockDim.y*blockIdx.y);
    int x = threadIdx.x + (blockDim.x*blockIdx.x);    

    float sum = 0.0f;

    if(y < numRows && x < numCols){
        cache[threadIdx.y*BLOCK_SIZE + threadIdx.x] = d_image[y*numCols + x];
        __syncthreads();

        
        for(int k = 0; k < BLOCK_SIZE; k++){ 
            sum += dctMatrix[k*BLOCK_SIZE + threadIdx.x] * cache[threadIdx.y*BLOCK_SIZE + k];
        }
        
        __syncthreads(); 
        //intermediate results (T*A) = I
        cache[threadIdx.y*BLOCK_SIZE + threadIdx.x] = sum;

        __syncthreads();

        sum = 0.0f;

        for(int k = 0; k < BLOCK_SIZE; k++){
            sum += IdctMatrix[threadIdx.y*BLOCK_SIZE + k] * cache[k*BLOCK_SIZE + threadIdx.x];
        }

        __syncthreads();
        //final result (I*T')
        f_image[y * numCols + x] = sum; //corresponding image in frequency domain


    }    

}

//(T'*A)*T
__global__ void IDCT(int numRows, int numCols, float *f_image, float *r_image) {

    __shared__ float cache[BLOCK_SIZE*BLOCK_SIZE];  
    int y = threadIdx.y + (blockDim.y*blockIdx.y);
    int x = threadIdx.x + (blockDim.x*blockIdx.x);    

    float sum = 0.0f;

    if(y < numRows && x < numCols){
        cache[threadIdx.y*BLOCK_SIZE + threadIdx.x] = f_image[y*numCols + x];
        __syncthreads();

        
        for(int k = 0; k < BLOCK_SIZE; k++){ 
            sum += IdctMatrix[k*BLOCK_SIZE + threadIdx.x] * cache[threadIdx.y*BLOCK_SIZE + k];
        }
        
        __syncthreads(); 
        //intermediate results (T*A) = I
        cache[threadIdx.y*BLOCK_SIZE + threadIdx.x] = sum;

        __syncthreads();

        sum = 0.0f;

        for(int k = 0; k < BLOCK_SIZE; k++){
            sum += dctMatrix[threadIdx.y*BLOCK_SIZE + k] * cache[k*BLOCK_SIZE + threadIdx.x];
        }

        __syncthreads();
        //final result (I*T')
        r_image[y * numCols + x] = sum; //restored image in spatial domain


    }    

}  

   
void compress(const int row, const int col, float *d_image, float *f_image)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksPerGrid(ceil(row/(float)threadsPerBlock.x), ceil(col/(float)threadsPerBlock.y), 1);
    DCT<<<blocksPerGrid, threadsPerBlock>>>(row, col, d_image, f_image);

}

void decompress(const int row, const int col, float *f_image, float *r_image)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksPerGrid(ceil(row/(float)threadsPerBlock.x), ceil(col/(float)threadsPerBlock.y), 1);
    IDCT<<<blocksPerGrid, threadsPerBlock>>>(row, col, f_image, r_image);

}

