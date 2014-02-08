#include    <wb.h>

#define BLOCK_SIZE 512

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Returns sum of input list of length n.
// lst[0] + lst[1] + ... + lst[n-1];
__global__ void total(float * input, float * output, int len) {

	__shared__ float partialSum[2 * BLOCK_SIZE];	
	
    // Load a segment of the input vector into shared memory provided the segment
	// is inside the input array bounds.
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;
	
	unsigned int inputLocation1 = start + t;
	partialSum[t] = (inputLocation1 < len) ? input[inputLocation1] : 0;
	
	unsigned int inputLocation2 = start + blockDim.x + t;
	partialSum[blockDim.x + t] = (inputLocation2 < len) ? input[inputLocation2] : 0;
	
	//printf("block %d - thread %d : Loading input element %d and %d\n", blockIdx.x, threadIdx.x, inputLocation1, inputLocation2);
	
	// Traverse the reduction tree summing each node's elements
	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
		
		// Wait for all threads to finish the previous step because the final values are needed in the current step.
    	__syncthreads();
		
		// If the curren thread is active sum the reduction tree elements.
      	if(t < stride)
          partialSum[t] += partialSum[t + stride];
  	}
	
    // Write the computed sum of the block to the output vector.
	if (t == 0) {
		output[blockIdx.x] = partialSum[0];
	}
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    
	// Allocate enough memory for the input and output vectors
	int inputMemSize = numInputElements * sizeof(float);
	wbLog(TRACE, "Allocating ", inputMemSize, " bytes of memory for input.");
	wbCheck(cudaMalloc((void **) &deviceInput, inputMemSize));
	
	int outputMemSize = numOutputElements * sizeof(float);
	wbLog(TRACE, "Allocating ", outputMemSize, " bytes of memory for output.");
	wbCheck(cudaMalloc((void **) &deviceOutput, outputMemSize));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    
	// Copy input vector into device global memory.
	wbCheck(cudaMemcpy(deviceInput, hostInput, inputMemSize, cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
	
	// Initialize the grid and block dimensions
	dim3 DimGrid(numOutputElements);
	dim3 DimBlock(BLOCK_SIZE);

    wbTime_start(Compute, "Performing CUDA computation");
	
    // Launch the GPU Kernel here
	total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements); 
	
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
	
    // Copy the GPU memory back to the CPU
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, outputMemSize, cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_start(GPU, "Freeing GPU Memory");
    
	// Free device global memory.
	wbCheck(cudaFree(deviceInput));
	wbCheck(cudaFree(deviceOutput));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}