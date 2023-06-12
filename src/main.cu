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
    image.convertTo(image_float, CV_32F);
    float *d_image, *DCT_res, *IDCT_res;
   
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&DCT_res, imageSize);
    cudaMalloc((void**)&IDCT_res, imageSize);

    cudaDeviceSynchronize();    

    cudaMemcpy(d_image, image_float.ptr<float>(), imageSize, cudaMemcpyHostToDevice);

    printf("Testing");
    
    cudaDeviceSynchronize();  
    LaunchDCT(image.rows, image.cols, d_image, DCT_res);
    cudaDeviceSynchronize();
    LaunchIDCT(image.rows, image.cols, DCT_res, IDCT_res);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    float* h_outputImage = (float*)malloc(imageSize);
    cudaMemcpy(h_outputImage, IDCT_res, imageSize, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();   

    uint8_t* outputImage = (uint8_t*)malloc(image.rows*image.cols*sizeof(uint8_t));

    for(unsigned int i = 0; i < image.rows*image.cols; i++){
        outputImage[i] = static_cast<uint8_t>(h_outputImage[i]);
    }

    cv::namedWindow("Image Window", cv::WINDOW_NORMAL);
    cv::imshow("Image Window", image);

    cv::Mat resultImage(image.rows, image.cols, CV_8UC1, outputImage);
    cv::namedWindow("Resultant Image", cv::WINDOW_NORMAL);
    cv::imshow("Resultant Image", resultImage);
    cv::waitKey(0);
    

    free(h_outputImage);
    free(outputImage);
    cudaFree(d_image);
    cudaFree(result);


    // A_h = (float*) malloc( sizeof(float)*mat_sz ); //returns a char pointer or void pointer for malloc
    // //Since I want float cast to float
    // //size of float * number of elements    
    
    // for (unsigned int i=0; i < mat_sz; i++) { A_h[i] = (rand()%100)/100.00; }
    // //creates an Array A_h on the host size of size mat_sz with elements 0->0.99
    // B_h = (float*) malloc( sizeof(float)*mat_sz );
    // for (unsigned int i=0; i < mat_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    // C_h = (float*) malloc( sizeof(float)*mat_sz );

    // // stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matDim, matDim,
    //     matDim, matDim, matDim, matDim);

    // // Allocate device variables ----------------------------------------------

    // printf("Allocating device variables..."); fflush(stdout);
    // // startTime(&timer);

    // /*************************************************************************/    
    
    // //allocate input vectors in device memory of size number of elements in mat_size * sizeof each element
    
    // cudaMalloc(&A_d, sizeof(float)*mat_sz);
    // cudaMalloc(&B_d, sizeof(float)*mat_sz);
    // cudaMalloc(&C_d, sizeof(float)*mat_sz);

    // //INSERT CODE HERE

    // /*************************************************************************/
    // cudaDeviceSynchronize();
    // // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // // Copy host variables to device ------------------------------------------
    // printf("Copying data from host to device..."); fflush(stdout);
    // // startTime(&timer);
	
    // /*************************************************************************/
    // //INSERT CODE HERE

    // cudaMemcpy(A_d, A_h, sizeof(float)*mat_sz, cudaMemcpyHostToDevice);
    // cudaMemcpy(B_d, B_h, sizeof(float)*mat_sz, cudaMemcpyHostToDevice);
    // cudaMemcpy(C_d, C_h, sizeof(float)*mat_sz, cudaMemcpyHostToDevice);

    // /*************************************************************************/
    // cudaDeviceSynchronize();
    // // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // // Launch kernel using standard mat-add interface ---------------------------
    // printf("Launching kernel..."); fflush(stdout);
    // // startTime(&timer);
    // basicMatAdd(mat_sz, A_d, B_d, C_d);

    // cuda_ret = cudaDeviceSynchronize();
    // if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    // // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // // Copy device variables from host ----------------------------------------
    // printf("Copying data from device to host..."); fflush(stdout);
    // // startTime(&timer);

    // /*************************************************************************/
    // //INSERT CODE HERE
    
    // //copy results from device memory to host
    // cudaMemcpy(A_h, A_d, sizeof(float)*mat_sz, cudaMemcpyDeviceToHost);
    // cudaMemcpy(B_h, B_d, sizeof(float)*mat_sz, cudaMemcpyDeviceToHost);
    // cudaMemcpy(C_h, C_d, sizeof(float)*mat_sz, cudaMemcpyDeviceToHost);
    // /*************************************************************************/
    // cudaDeviceSynchronize();
    // // stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // // Verify correctness -----------------------------------------------------

    // printf("Verifying results...\n"); fflush(stdout);

    // // verify(A_h, B_h, C_h, matDim);


    // // Free memory ------------------------------------------------------------

    // free(A_h);
    // free(B_h);
    // free(C_h);

    // /*************************************************************************/
    // //INSERT CODE HERE

    // cudaFree(A_d);
    // cudaFree(B_d);
    // cudaFree(C_d);
    cv::destroyWindow("Image Window");
    cv::destroyWindow("Resultant Image");
    /*************************************************************************/
    return 0;
}

