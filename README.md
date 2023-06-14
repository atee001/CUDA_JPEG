# JPEG Compression and Decompression in Parallel
# Project Idea / Overview
The goal of this project is to implement JPEG compression and decompression from scratch (no libraries except openCV to display images) without Huffman encoding in parallel. 
The JPEG Compression and Decompression follows this block diagram:
![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/84dfd51c-5b03-4227-b622-96c7b590df3e)
Step 1. Divide the image into 8 x 8 non-overlapping blocks.  

Step 2. Take the Discrete Cosine Transform of each 8 x 8 block. This converts the image to the frequency domain. 
The higher frequencies are are the bottom right of each 8 x 8 block and the lower frequencies are the top left corner. 

Formal Equation for Discrete Cosine Transform:
![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/2268004f-1bbd-4264-9f60-893573a43d52)

Step 3. Quantization step: Remove high frequency components of each 8 x 8 block using a mask. Give user the choice of how many DCT coefficients they want to keep.
We can do this because the Human Visual System is less sensitive to changes in the High frequencies. 

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/474044e6-9a72-439b-ab65-a90c514b10b3)

Step 4. Take the Inverse Discrete Cosine Transform of each 8 x 8 block. This converts the image back to the spatial domain without visually redundant information.

Formal Equation for Inverse Discrete Cosine Transform:
![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/762603ba-e06a-4c15-8678-add4157c341f)

#How is the GPU used to accelerate the application?

The Discrete Cosine Transform and Inverse Discrete Cosine Transform of each 8 x 8 block can be computed in parallel as each block is independent each other.

DCT can be computed using Matrix Multiplication following this equation:

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/e6a93c1f-a053-4529-bf98-15e047126cc8)

Where C Matrix value is:

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/154f340b-8a6e-4b5e-83f4-507fec5f8777)

Where Transposed C Matrix value is:

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/07a0d672-080e-469a-9cb7-772dcebdf7ef)

To compute the Discrete Cosine Transform I perform tiled Matrix Multiplication.

1. I store each 8 x 8 sub-image into shared memory.

2. I store C and Transposed C Matrix into Constant Memory. 

3. I compute Matrix Multiplication of C * X. 


Quantization can also be applied in each 8 x 8 Block in parallel to remove high frequency components.
Shared Memory is used for 8 x 8 Matrix Multiplication to improve memory access speed.
Constant Memory is used to store the DCT Matrix and Transposed DCT Matrix as these coefficients are constant. 

Implementation details
Documentation on how to run your code
Evaluation/Results
Problems faced
On the last page, include a table with a list of tasks for the project, and a percentage breakdown of contribution of each team member for the tasks. You can choose the granularity of task breakdown here.

Implemented JPEG Compression and Decompression in CUDA.
Implemented Discrete Cosine Transform, Inverse Discrete Transform, Zonal Coding from scratch and in parallel. 
Utilized shared memory to do matrix multiplication for DCT. 
Utilized constant memory to store DCT predefined 8 x 8 Matrix. 
Used openCV to display the images.
Demo Video: https://youtu.be/WkKCmrhht1o
