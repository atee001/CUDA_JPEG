#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"

int main (int argc, char *argv[])
{

    // Timer timer;
    cudaError_t cuda_ret;
    // Initialize host variables ----------------------------------------------
    cv::Mat image = cv::imread("/home/eemaj/atee/ee147/jpeg/CUDA_JPEG/images/lena_std.tif");    
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

    // float *A_h, *B_h, *C_h;
    // float *A_d, *B_d, *C_d;
    // size_t mat_sz;
    // unsigned matDim;
    // dim3 dim_grid, dim_block;
    size_t imageSize = image.rows*image.cols*sizeof(float);

    if (argc == 1) {
        imageSize = image.rows*image.cols*sizeof(float);
    } 
    else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./mat-add                # All Images are 512 x 512"
      "\n");
        exit(0);
    }
    cv::Mat image_float; 
    image.convertTo(image_float, CV_32F, 1.0/255);
    float *d_image, *DCT_res, *IDCT_res, *temp, *result;
   
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&DCT_res, imageSize);
    cudaMalloc((void**)&IDCT_res, imageSize);
    cudaMalloc((void**)&temp, imageSize);
    cudaMalloc((void**)&result, imageSize);

    cudaDeviceSynchronize();    

    cudaMemcpy(d_image, image_float.ptr<float>(), imageSize, cudaMemcpyHostToDevice);

    printf("Testing");
    
    cudaDeviceSynchronize();  
    LaunchDCT(image.rows, image.cols, d_image, DCT_res);
    cudaDeviceSynchronize();
    LaunchIDCT(image.rows, image.cols, DCT_res, IDCT_res);
    cudaDeviceSynchronize();
    LaunchIDCT(image.rows, image.cols, IDCT_res, temp);
    cudaDeviceSynchronize();
    LaunchDCT(image.rows, image.cols, temp, result);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    float* h_outputImage = (float*)malloc(imageSize);
    cudaMemcpy(h_outputImage, result, imageSize, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();   

    for (unsigned int i = 0; i < image.rows * image.cols; i++) {
        h_outputImage[i] *= 255.0;
    }

    // Convert the matrix to CV_8U data type
    cv::Mat resultImage(image.rows, image.cols, CV_8U);
    for (int i = 0; i < resultImage.rows; i++) {
        for (int j = 0; j < resultImage.cols; j++) {
            resultImage.at<uint8_t>(i, j) = static_cast<uint8_t>(h_outputImage[i * resultImage.cols + j]);
        }
    }
    
    

    cv::namedWindow("Image Window", cv::WINDOW_NORMAL);
    cv::imshow("Image Window", image);
    cv::namedWindow("Resultant Image", cv::WINDOW_NORMAL);
    cv::imshow("Resultant Image", resultImage);   
    cv::waitKey(0);
    

    free(h_outputImage);
    // free(outputImage);
    cudaFree(d_image);
    cudaFree(DCT_res);
    cudaFree(IDCT_res);


    // cv::destroyWindow("Image Window");
    cv::destroyWindow("Resultant Image");
    /*************************************************************************/
    return 0;
}

