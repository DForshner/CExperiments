#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


#define Mask_width  5
#define Mask_radius Mask_width/2
#define O_TILE_WIDTH 16
#define BLOCK_WIDTH (O_TILE_WIDTH + Mask_width - 1)

__device__ inline float clamp(float val) {
	return min(max(val, 0.0), 1.0);
}

__global__ void convolution2D(float* I, const float* __restrict__ M, float* P,
							  int channels, int width, int height) {
	
	__shared__ float N_ds[BLOCK_WIDTH][BLOCK_WIDTH];
	
	// For each color channel
	for (int channel = 0; channel < channels; ++channel) {
	
		int row_o = blockIdx.y * O_TILE_WIDTH + threadIdx.y;
		int col_o = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
		
		// Input elements are shifted by the mask radius compared to output
		int row_i = row_o - Mask_radius;
		int col_i = col_o - Mask_radius;

		// Copy the input element into shared memory if inside the image's boundary.
		// Otherwise assign ghost value (0)
		if ((row_i >= 0) && (row_i < height) &&
			(col_i >= 0) && (col_i < width)) {
			
			// Determine linerized input address
			int element_i = row_i * width + col_i;
			int source = element_i * channels + channel;
			
			N_ds[threadIdx.y][threadIdx.x] = I[source];
		} else {			
			N_ds[threadIdx.y][threadIdx.x] = 0.0f;
		}
		
		// Wait for all threads to finish copying input data into shared memory.
		__syncthreads();

		// Calculate thread's output element if the output element is is inside the
		// current output tile range.
		float value = 0.0f;
		if (threadIdx.x < O_TILE_WIDTH && threadIdx.y < O_TILE_WIDTH) {
			for (int row_m = 0; row_m < Mask_width; row_m++) {
				for (int col_m = 0; col_m < Mask_width; col_m++) {
					value += M[row_m * Mask_width + col_m] * N_ds[threadIdx.y + row_m][threadIdx.x + col_m]; 
				}
			}
		}
		
		// Copy output element into global memory if inside the image's boundary.
		if (threadIdx.x < O_TILE_WIDTH && threadIdx.y < O_TILE_WIDTH) {
			// Determine linerized output address
			int element_o = row_o * width + col_o;
			int destination = element_o * channels + channel;
			
			P[destination] = clamp(value);
			assert(P[destination] >= 0 && P[destination] <= 1);
		}
		
		// Wait for all threads to finish copying output data into global memory.
		__syncthreads();
	}
}

int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
	
	dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    wbLog(TRACE, "blockDim is ", BLOCK_WIDTH, ",", BLOCK_WIDTH);
	
	wbLog(TRACE, "Image is ", imageWidth, " x ", imageHeight);
	
	int gridWidth = (imageWidth - 1) / O_TILE_WIDTH + 1; // Ceiling
	int gridHeight = (imageHeight - 1) / O_TILE_WIDTH + 1; // Ceiling
	dim3 gridDim(gridWidth, gridHeight);
	wbLog(TRACE, "gridDim is ", gridWidth, ",", gridHeight);

	convolution2D<<<gridDim, blockDim>>>(deviceInputImageData, deviceMaskData,
										deviceOutputImageData, imageChannels, 
										 imageWidth, imageHeight);
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}