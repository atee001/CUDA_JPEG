#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#define BLOCK_SIZE 8
#define FILTER_SIZE 3

using namespace cv;

//DCT matrix T obtained from matlab dctmtx(8)
__constant__ double dctMatrix[FILTER_SIZE * FILTER_SIZE] = {
	-1.0, 0.0, 1.0,
	-2.0, 0.0, 2.0,
	-1.0, 0.0, 1.0
};

//transposed DCT matrix T' obtained from matlab dctmtx(8) with a transpose

__global__ void DCT(int numRows, int numCols, double *d_image, double *result_image) {

	int col = threadIdx.x + (blockDim.x * blockIdx.x);
	int row = threadIdx.y + (blockDim.y * blockIdx.y);
	
	if(row < numRows && col < numCols)
	{
		int startX = col - (FILTER_SIZE/2);
		int startY = row - (FILTER_SIZE/2);
		double temp = 0.0;

		for(int i = 0; i < numRows; i++)
		{
			for(int j = 0; j < numCols; j++)
			{
				if(( (startX + i >= 0) && (startX + i < numCols) ) && ( (startY + i >= 0) && (startY + i < numRows)))
				{
					
					temp += d_image[numCols*(startY + i) + startX + i] * dctMatrix[i*FILTER_SIZE + j];	
				}			

			}
		}
	
		result_image[row*numCols + col] = temp;
	}

}

//(T'*A)*T
void compress(const int numRows, const int numCols, double *d_image, double* result_image)
{

    dim3 threadsPerBlock(32, 32, 1);
    dim3 blocksPerGrid(ceil(numCols/(double)threadsPerBlock.x), ceil(numRows/(double)threadsPerBlock.y), 1);
    DCT<<<blocksPerGrid, threadsPerBlock>>>(numRows, numCols, d_image, result_image);

}


