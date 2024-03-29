# JPEG Compression and Decompression in Parallel
# Team Members: Andrew Tee
# Demo Video: https://youtu.be/WkKCmrhht1o
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

# How is the GPU used to accelerate the application?

The Discrete Cosine Transform and Inverse Discrete Cosine Transform of each 8 x 8 block can be computed in parallel as each block is independent each other.

DCT and Inverse DCT can be computed using Matrix Multiplication following this equation where D is the DCT Matrix, A is the 8 x 8 Sub-Image, D' is the Transposed DCT Matrix.

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/341f5d99-9973-4022-8dc3-a8e4c40e6cad)

Where D Matrix value is:

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/154f340b-8a6e-4b5e-83f4-507fec5f8777)

Where D' Matrix value is:

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/07a0d672-080e-469a-9cb7-772dcebdf7ef)

To compute the Discrete Cosine Transform I perform tiled Matrix Multiplication.

I launch the kernel using 8 x 8 block size (since each sub-image is 8 x 8) and Gridsize of ceil((Image Rows)/8) && ceil((Image Cols)/8). 

1. I store each 8 x 8 sub-image into shared memory.

2. I store D and Transposed D Matrix into Constant Memory. 

3. I compute Matrix Multiplication of D * A and store the results back into shared memory (A is 8 x 8 sub-image).

4. I then Compute (D * A)*D' by Matrix Multiplying the intermediate results in (3) by D'. (Now Image is in Frequency Domain).

To Quantize and discard high frequency components.

1. I store the User Defined Filter in shared memory. 

2. I then elementwise multiply each 8 x 8 sub-image (Frequency Domain) by the User Defined Filter. 

3. Now the Image is Compressed.

To Decompress I perform the Inverse Discrete Cosine Transform using tiled Matrix Multiplication. 

1. I store each 8 x 8 sub-image in frequency domain into shared memory.

2. I compute Matrix Multiplication of D' * A and store the results back into shared memory.

3. I then Compute (D' * A)* D by Matrix Multiplying the intermediate results in (3) by D. (Now Image is in Spatial Domain).

# Documentation/Results

___Important!! Change the absolute path for the image file in the main code near line 131:___

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/39a0298d-a493-479d-8467-756cb2daa270)

The main code resizes the input image into 512 x 512 and Grayscale. This avoids using zero padding in case the Image rows and columns are not evenly divisibly by 8. 

To run the code type the following commands:

make

./jpeg

The Terminal will prompt a mask size for how many coefficients to be kept enter an option from (1-5):

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/eeece8bf-ae73-4220-98ed-5aa5c4953335)

Once the User selects a mask size the Original, Frequency Domain, and Decompressed Image is shown (Keeping all DCT 64 Coefficients)

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/5ffbfe24-66f5-4dbc-82f4-e288f5bf3529)

Lena Example (Keeping 32 DCT Coefficients)

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/9e854a0a-5ddd-496f-8735-1d9ab677cdd2)

Lena Example (Keeping 15 DCT Coefficients)

![image](https://github.com/atee001/CUDA_JPEG/assets/80326381/d90ce593-d79a-4f99-9af5-798458c3749c)

The results are as expected the program correctly implements JPEG Compression without huffman encoding and decoding. The program is able to compute DCT, Quantization, and IDCT in parallel. 

# Problems Faced

The kernel block size is 8 x 8 therefore the hardware is underutilized as well as DRAM Burst. 

A better way to do this is to divide the image into multiple 32 x 32 sub-images and compute the 8 x 8 DCT within a 32 x 32 sub image. Launch the kernel with 32 x 32 block size however indexing this would be harder.

Kept facing indexing problems had to compare DCT and IDCT matrix multiplication results so I Had to Implement DCT and IDCT in Matlab to validate the results were correct. 

Kept using the wrong index for element wise multiplication with Filter. Eventually figured out to use mod 8 since the filter repeats every 8 columns and 8 rows. 

Couldn't see the Image in frequency as the pixel values were too small. Had to expand the dynamic range using Log of abs of image. Otherwise the entire frequency domain image was black. 

Task	Breakdown

Implementation of Discrete Cosine Transform in parallel: Andrew Tee - 100%

Implementation of Quantization in parallel using user defined filters: Andrew Tee - 100%

Implementation of Inverse Discrete Cosine Transform in parallel: Andrew Tee - 100%

Using OpenCV to display results and driver code: Andrew Tee - 100%

Project Report - Andrew Tee - 100%









