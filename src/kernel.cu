#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#define BLOCK_SIZE 8

using namespace cv;

//DCT matrix T obtained from matlab dctmtx(8)
__constant__ double dctMatrix[BLOCK_SIZE * BLOCK_SIZE] = {
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
__constant__ double IdctMatrix[BLOCK_SIZE * BLOCK_SIZE] = {
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

__global__ void DCT(int numRows, int numCols, double *d_image, double *f_image, double *zonalFilter) {

    __shared__ double cache[BLOCK_SIZE*BLOCK_SIZE];  
    __shared__ double filter[BLOCK_SIZE*BLOCK_SIZE];
    int y = threadIdx.y + (blockDim.y*blockIdx.y);
    int x = threadIdx.x + (blockDim.x*blockIdx.x); 
    double sum = 0.0;

    if(y < numRows && x < numCols){
        cache[threadIdx.y*BLOCK_SIZE + threadIdx.x] = d_image[y*numCols + x];
        filter[threadIdx.y*BLOCK_SIZE + threadIdx.x] = zonalFilter[threadIdx.y*BLOCK_SIZE + threadIdx.x];
        __syncthreads();

        
        for(int k = 0; k < BLOCK_SIZE; k++){ 
            sum += dctMatrix[k*BLOCK_SIZE + threadIdx.x] * cache[threadIdx.y*BLOCK_SIZE + k];
        }
        
        __syncthreads(); 
        //intermediate results (T*A) = I
        cache[threadIdx.y*BLOCK_SIZE + threadIdx.x] = sum;

        __syncthreads();

        sum = 0.0;
        for(int k = 0; k < BLOCK_SIZE; k++){
            sum += IdctMatrix[threadIdx.y*BLOCK_SIZE + k] * cache[k*BLOCK_SIZE + threadIdx.x];
        }

        __syncthreads();
        //final result (I*T')
        f_image[y * numCols + x] = sum; //corresponding image in frequency domain
        
        __syncthreads();
        //elemnent wise multiply with the filter
        //every 8 is 0 
        f_image[y * numCols + x] *= filter[((y % 8)*8) + (x % 8)];
    }    
}

//(T'*A)*T
__global__ void IDCT(int numRows, int numCols, double *f_image, double *r_image) {

    __shared__ double cache[BLOCK_SIZE*BLOCK_SIZE];  
    int y = threadIdx.y + (blockDim.y*blockIdx.y);
    int x = threadIdx.x + (blockDim.x*blockIdx.x);    

    double sum = 0.0;

    if(y < numRows && x < numCols){
        cache[threadIdx.y*BLOCK_SIZE + threadIdx.x] = f_image[y*numCols + x];
        __syncthreads();

        
        for(int k = 0; k < BLOCK_SIZE; k++){ 
            sum += IdctMatrix[k*BLOCK_SIZE + threadIdx.x] * cache[threadIdx.y*BLOCK_SIZE + k];
        }
        
        __syncthreads(); 
        //intermediate results (T'*F_image) = temp
        cache[threadIdx.y*BLOCK_SIZE + threadIdx.x] = sum;

        __syncthreads();

        sum = 0.0;
        //temp * T multiply rows of cache with columns of T
        for(int k = 0; k < BLOCK_SIZE; k++){
            sum += dctMatrix[threadIdx.y*BLOCK_SIZE + k] * cache[k*BLOCK_SIZE + threadIdx.x];
        }

        __syncthreads();
        //final result (I*T')
        r_image[y * numCols + x] = sum; //restored image in spatial domain


    }    

}  
   
void compress(const int numRows, const int numCols, double *d_image, double *f_image, double *zonalFilter)
{

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksPerGrid(ceil(numCols/(double)threadsPerBlock.x), ceil(numRows/(double)threadsPerBlock.y), 1);
    DCT<<<blocksPerGrid, threadsPerBlock>>>(numRows, numCols, d_image, f_image, zonalFilter);

}

void decompress(const int numRows, const int numCols, double *f_image, double *r_image)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocksPerGrid(ceil(numCols/(double)threadsPerBlock.x), ceil(numRows/(double)threadsPerBlock.y), 1);
    IDCT<<<blocksPerGrid, threadsPerBlock>>>(numRows, numCols, f_image, r_image);

}

