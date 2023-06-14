#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"

double calculateMSE(const cv::Mat& image1, const cv::Mat& image2) {

    cv::Mat diff;
    cv::absdiff(image1, image2, diff); //absolute value difference    
    cv::Mat squaredDiff = diff.mul(diff); // Calculate the squared difference    
    cv::Scalar mse = cv::mean(squaredDiff); // Calculate the mean squared error
    double mseValue = mse[0];  // single channel scaler since grayscale image
    return mseValue/(image1.rows*image1.cols);
}

cv::Mat createZonalFilter15()
{
    cv::Mat zonalFilter = cv::Mat::zeros(8, 8, CV_64F);
    double maskData[8][8] = {
        1, 1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };
    memcpy(zonalFilter.data, maskData, sizeof(maskData));
    return zonalFilter;
}

cv::Mat createZonalFilter32()
{
    cv::Mat zonalFilter = cv::Mat::zeros(8, 8, CV_64F);
    double maskData[8][8] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };
    memcpy(zonalFilter.data, maskData, sizeof(maskData));
    return zonalFilter;
}

cv::Mat createZonalFilterAll()
{
    cv::Mat zonalFilter = cv::Mat::zeros(8, 8, CV_64F);
    double maskData[8][8] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1
    };
    memcpy(zonalFilter.data, maskData, sizeof(maskData));
    return zonalFilter;
}

int main (int argc, char *argv[])
{
    int choice;
    
    printf("Choose the zonal filter:\n");
    printf("1. Retain 15 coefficients\n");
    printf("2. Retain 32 coefficients\n");
    printf("3. Retain all coefficients\n");
    printf("Enter your choice (1-3): ");
    scanf("%d", &choice);

    cv::Mat zonalFilter;

    switch (choice){
    case 1:
        zonalFilter = createZonalFilter15();
        break;
    case 2:
        zonalFilter = createZonalFilter32();
        break;
    case 3:
        zonalFilter = createZonalFilterAll();
        break;
    default:
        printf("Invalid choice!\n");
        return -1;
    }

    printf("Selected Zonal Filter:\n");
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("%.0f ", zonalFilter.at<double>(i, j));
        }
        printf("\n");
    }

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

    // double *A_h, *B_h, *C_h;
    // double *A_d, *B_d, *C_d;
    // size_t mat_sz;
    // unsigned matDim;
    // dim3 dim_grid, dim_block;
    size_t imageSize = image.rows*image.cols*sizeof(double);

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
    double *d_image, *f_image, *r_image;
   
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&f_image, imageSize);
    cudaMalloc((void**)&r_image, imageSize);

    cudaDeviceSynchronize();    

    cudaMemcpy(d_image, image_double.ptr<double>(), imageSize, cudaMemcpyHostToDevice);

    printf("Testing");
    
    cudaDeviceSynchronize();  

    compress(image.rows, image.cols, d_image, f_image); //returns image in frequency domain
    cudaDeviceSynchronize();
    decompress(image.rows, image.cols, f_image, r_image); //returns image in spatial domain
    // cudaDeviceSynchronize();
    // LaunchIDCT(image.rows, image.cols, IDCT_res, temp);
    // cudaDeviceSynchronize();
    // LaunchDCT(image.rows, image.cols, temp, result);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    double* h_outputImage = (double*)malloc(imageSize);
    double* f_outputImage = (double*)malloc(imageSize);
    cudaMemcpy(h_outputImage, r_image, imageSize, cudaMemcpyDeviceToHost);    
    cudaMemcpy(f_outputImage, f_image, imageSize, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();   

    // for (unsigned int i = 0; i < image.rows * image.cols; i++) {
    //     h_outputImage[i] *= 255.0;
    // }
    

    // Convert the matrix to CV_8U data type


    cv::Mat resultImage(image.rows, image.cols, CV_64F);
    memcpy(resultImage.data, h_outputImage, imageSize);
    cv::normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    resultImage.convertTo(resultImage, CV_8U);
    // for (int i = 0; i < resultImage.rows; i++) {
    //     for (int j = 0; j < resultImage.cols; j++) {
    //         resultImage.at<uint8_t>(i, j) = static_cast<uint8_t>(h_outputImage[i * resultImage.cols + j]);
    //     }
    // }

    cv::Mat frequencyImage(image.rows, image.cols, CV_64F);
    
    memcpy(frequencyImage.data, f_outputImage, imageSize);

    cv::log(cv::abs(frequencyImage) + 1, frequencyImage);
    cv::normalize(frequencyImage, frequencyImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    // frequencyImage.convertTo(frequencyImage, CV_8U);
    cv::namedWindow("Frequency Image", cv::WINDOW_NORMAL);
    cv::imshow("Frequency Image", frequencyImage);    
    // cv::normalize(resultImage, resultImage, 0, 255, cv::NORM_MINMAX, CV_8U);
    

    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::imshow("Original Image", image);
    cv::namedWindow("Decompressed Image", cv::WINDOW_NORMAL);
    cv::imshow("Decompressed Image", resultImage);   
    cv::waitKey(0);
    

    free(h_outputImage);
    free(f_outputImage);
    // free(outputImage);
    cudaFree(d_image);
    cudaFree(f_image);
    cudaFree(r_image);


    // cv::destroyWindow("Image Window");
    cv::destroyWindow("Decompressed Image");
    cv::destroyWindow("Original Image");
    cv::destroyWindow("Frequency Image");

    double mse = calculateMSE(image, resultImage);
    std::cout << "MSE of Original Image Vs Decompressed Image: " << mse << std::endl;



    /*************************************************************************/
    return 0;
}

