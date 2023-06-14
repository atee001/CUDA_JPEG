# CUDA_JPEG
# Project Idea / Overview
The goal of this project is to implement JPEG compression and decompression from scratch (no cuFFT) without Huffman encoding in parallel. 
The JPEG Compression and Decompression follows this block diagram:
![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/84dfd51c-5b03-4227-b622-96c7b590df3e)
Step 1. Divide the image into 8 x 8 blocks 

Step 2. Take the Discrete Cosine Transform of each 8 x 8 block. This converts the image to the frequency domain. 
The higher frequencies are are the bottom right of each 8 x 8 block and the lower frequencies are the top left corner. 
We can do this because the Human Visual System is less sensitive to changes in the High frequencies. 
Formal Equation for Discrete Cosine Transform:
![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/2268004f-1bbd-4264-9f60-893573a43d52)

Step 3. Remove high frequency components of each 8 x 8 block using a mask. Give user the choice of how many DCT coefficients they want to keep.

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/474044e6-9a72-439b-ab65-a90c514b10b3)

Step 4. Take the Inverse Discrete Cosine Transform of each 8 x 8 block. This converts the image back to the spatial domain without visually redundant information.
Formal Equation for Inverse Discrete Cosine Transform:
![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/762603ba-e06a-4c15-8678-add4157c341f)

How is the GPU used to accelerate the application?
The Discrete Cosine Transform of each 8 x 8 block can be computed in parallel.
Each 8 x 8 block is independent of each other.




Examples of this includes:
Details related to the parallel algorithm
How is the problem space partitioned into threads / thread blocks
Which stage of the software pipeline is parallelized
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
