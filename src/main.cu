#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"


int main (int argc, char *argv[])
{
    cudaError_t cuda_ret;
    // Initialize host variables ----------------------------------------------
    
    //please replace the full path of the image
    cv::Mat image = cv::imread("/home/andrtee1/cudaPractice/CUDA_JPEG/images/lena_std.tif");    
    if (image.empty())
    {
        printf("Failed to read image exitting...");
        return 1;
    }

    //preprocess convert image to 512 x 512 and single channel

    cv::resize(image, image, cv::Size(512, 512));
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);    

    printf("\nSetting up the problem..."); fflush(stdout);
    // startTime(&timer);

    // double *A_h, *B_h, *C_h;
    // double *A_d, *B_d, *C_d;
    // size_t mat_sz;
    // unsigned matDim;
    // dim3 dim_grid, dim_block;
    size_t imageSize = image.rows*image.cols*sizeof(double);
    size_t filterSize = 8*8*sizeof(double);
    if (argc == 1) {
        imageSize = image.rows*image.cols*sizeof(double);
    } 
    else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./mat-add                # All Images are 512 x 512"
      "\n");
        exit(0);
    }
    cv::Mat image_double; 
    image.convertTo(image_double, CV_64F);
    double *d_image;
    double *result_image;
   
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&result_image, imageSize);

    cudaDeviceSynchronize();    

    cudaMemcpy(d_image, image_double.ptr<double>(), imageSize, cudaMemcpyHostToDevice);
    // printf("Testing");
    
    cudaDeviceSynchronize();  
    //call kernel here  
    compress(512, 512, d_image, result_image);
    cudaDeviceSynchronize();
    // cudaDeviceSynchronize();
    // LaunchIDCT(image.rows, image.cols, IDCT_res, temp);
    // cudaDeviceSynchronize();
    // LaunchDCT(image.rows, image.cols, temp, result);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    double* h_outputImage = (double*)malloc(imageSize);
    cudaMemcpy(h_outputImage, result_image, imageSize, cudaMemcpyDeviceToHost);    

    cudaDeviceSynchronize();   

    // for (unsigned int i = 0; i < image.rows * image.cols; i++) {
    //     h_outputImage[i] *= 255.0;
    // }
        // printf("Testing");

    // Convert the matrix to CV_8U data type


    cv::Mat resultImage(image.rows, image.cols, CV_64F);
    memcpy(resultImage.data, h_outputImage, imageSize);
    //cv::normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    resultImage.convertTo(resultImage, CV_8U);
    cv::imwrite("edgeImage.jpg", resultImage);
	
   // cv::namedWindow("Filtered Image", cv::WINDOW_NORMAL);
    //cv::imshow("Filtered Image", resultImage);    

    //cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    image.convertTo(image, CV_8U);
    //cv::imshow("Original Image", image);
    //cv::waitKey(0);
  

    free(h_outputImage);
    // free(outputImage);
    cudaFree(d_image);
    // cv::destroyWindow("Image Window");
  //  cv::destroyWindow("Original Image");
    //cv::destroyWindow("Frequency Image");
    /*************************************************************************/


    return 0;
}

