#include	<wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
	register int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// C[i] = A[i] + B[i]
	
	if (i < len)
		out[i] = in1[i] + in2[i];
}

static const unsigned THREADS = 512;

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
	
    // Allocate GPU memory	
	size_t size = inputLength * sizeof(float);
	wbLog(TRACE, "Element size is: ", size);
	
	wbCheck(cudaMalloc((void **) &deviceInput1, size));
	wbCheck(cudaMalloc((void **) &deviceInput2, size));
	wbCheck(cudaMalloc((void **) &deviceOutput, size));
	
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
	
    // Copy memory to the GPU
	cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    // Initialize the grid and block dimensions
	dim3 DimGrid((inputLength - 1)/THREADS + 1, 1, 1);
	dim3 DimBlock(THREADS, 1, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");
	
    //Launch the GPU Kernel
	vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength); 
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
	
    //Copy the GPU memory back to the CPU
	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
	
    // Free the GPU memory
	wbCheck(cudaFree(deviceInput1));
	wbCheck(cudaFree(deviceInput2));
	wbCheck(cudaFree(deviceOutput));
	
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}