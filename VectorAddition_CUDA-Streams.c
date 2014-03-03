#include	<wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void vecAdd(float * in1, float * in2, float * out, int len, int streamId, int startIdx) {
	
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	if ((i + startIdx) < len) {
		out[i] = in1[i] + in2[i];
		//printf("\n[%d - %d - %d] [%d] -> (%d / %d) -> %f + %f = %f", 
		//	   blockIdx.x, threadIdx.x, streamId, i, (startIdx + i), len, in1[i], in2[i], out[i]);		
	}
}

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
	
	int segSize = 256;
	int segByteSize = segSize * sizeof(float);
	wbLog(TRACE, "The segment size is:", segSize, " elements ", segByteSize, " bytes.");
	
	wbTime_start(GPU, "Allocating Streams.");
	
	cudaStream_t stream0, stream1, stream2, stream3;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	
    wbTime_stop(GPU, "Allocating Streams.");
	
	wbTime_start(GPU, "Allocating GPU memory.");
	
	size_t size = inputLength * sizeof(float);
	wbLog(TRACE, "Input is: ", inputLength, " elements ", size, " bytes.");
	
	float *d_A0, *d_B0, *d_C0;
	float *d_A1, *d_B1, *d_C1;
	float *d_A2, *d_B2, *d_C2;
	float *d_A3, *d_B3, *d_C3;
	
	wbCheck(cudaMalloc((void **) &d_A0, size));
	wbCheck(cudaMalloc((void **) &d_B0, size));
	wbCheck(cudaMalloc((void **) &d_C0, size));
	
	wbCheck(cudaMalloc((void **) &d_A1, size));
	wbCheck(cudaMalloc((void **) &d_B1, size));
	wbCheck(cudaMalloc((void **) &d_C1, size));
	
	wbCheck(cudaMalloc((void **) &d_A2, size));
	wbCheck(cudaMalloc((void **) &d_B2, size));
	wbCheck(cudaMalloc((void **) &d_C2, size));
	
	wbCheck(cudaMalloc((void **) &d_A3, size));
	wbCheck(cudaMalloc((void **) &d_B3, size));
	wbCheck(cudaMalloc((void **) &d_C3, size));
	
    wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Allocating pinned host memory.");
	
	float *h_A, *h_B, *h_C;
	cudaHostAlloc((void**) &h_A, size, cudaHostAllocDefault);
	cudaHostAlloc((void**) &h_B, size, cudaHostAllocDefault);
	cudaHostAlloc((void**) &h_C, size, cudaHostAllocDefault);
	
	wbTime_stop(GPU, "Allocating pinned host memory.");
	
	wbTime_start(GPU, "Copying to pinned host memory.");
	
	memcpy(h_A, hostInput1, size);
	assert(h_A[0] == hostInput1[0]);
	memcpy(h_B, hostInput2, size);
	assert(h_B[0] == hostInput2[0]);
		
	//printf("\ninput[0] is %f + %f", h_A[0], h_B[0]);
	//printf("\ninput[256] is %f + %f", h_A[256], h_B[256]);
	//printf("\ninput[512] is %f + %f", h_A[512], h_B[512]);
	//printf("\ninput[768] is %f + %f", h_A[768], h_B[768]);
	
	wbTime_stop(GPU, "Copying to pinned host memory.");
	
	wbTime_start(GPU, "Computing results.");
	
	for (int i = 0; i < inputLength ; i += segSize * 4) {
		//printf("\n################################ - Iteration %d", i / (segSize * 4));
		
		int stream0Size = ((i + segSize * 1 - 1) < inputLength) ? segSize : (inputLength - (i + segSize * 0));
		if (stream0Size < 0) stream0Size = 0;
	
		int stream1Size = ((i + segSize * 2 - 1) < inputLength) ? segSize : (inputLength - (i + segSize * 1));
		if (stream1Size < 0) stream1Size = 0;

		int stream2Size = ((i + segSize * 3 - 1) < inputLength) ? segSize : (inputLength - (i + segSize * 2));
		if (stream2Size < 0) stream2Size = 0;
	
		int stream3Size = ((i + segSize * 4 - 1) < inputLength) ? segSize : (inputLength - (i + segSize * 3));
		if (stream3Size < 0) stream3Size = 0;
	
		// Copy from pinned host to device memory.
		
		cudaMemcpyAsync(d_A0, (h_A + i + segSize * 0), stream0Size * sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, (h_B + i + segSize * 0), stream0Size * sizeof(float), cudaMemcpyHostToDevice, stream0);
		//printf("\n#### - copy elements starting from %d to %d to stream 0", (i + segSize * 0), (i + segSize * 0) + stream0Size);
		
		cudaMemcpyAsync(d_A1, (h_A + i + segSize * 1), stream1Size * sizeof(float), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(d_B1, (h_B + i + segSize * 1), stream1Size * sizeof(float), cudaMemcpyHostToDevice, stream1);
		//printf("\n#### - copy elements starting from %d to %d to stream 1", (i + segSize * 1), (i + segSize * 1) + stream1Size);
		
		cudaMemcpyAsync(d_A2, (h_A + i + segSize * 2), stream2Size * sizeof(float), cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(d_B2, (h_B + i + segSize * 2), stream2Size * sizeof(float), cudaMemcpyHostToDevice, stream2);
		//printf("\n#### - copy elements starting from %d to %d to stream 2", (i + segSize * 2), (i + segSize * 2) + stream2Size);
		
		cudaMemcpyAsync(d_A3, (h_A + i + segSize * 3), stream3Size * sizeof(float), cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(d_B3, (h_B + i + segSize * 3), stream3Size * sizeof(float), cudaMemcpyHostToDevice, stream3);
		//printf("\n#### - copy elements starting from %d to %d to stream 3", (i + segSize * 3), (i + segSize * 3) + stream3Size);

		// Wait for all streams to finish copying to device.
		cudaDeviceSynchronize();
		
		// Compute iteration's segments.
		vecAdd<<<segSize/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, inputLength, 0, (i + segSize * 0));
		vecAdd<<<segSize/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, inputLength, 1, (i + segSize * 1));
		vecAdd<<<segSize/256, 256, 0, stream2>>>(d_A2, d_B2, d_C2, inputLength, 2, (i + segSize * 2));
		vecAdd<<<segSize/256, 256, 0, stream3>>>(d_A3, d_B3, d_C3, inputLength, 3, (i + segSize * 3));
		
		// Wait for all streams to finish computation.
		cudaDeviceSynchronize();
		
		// Copy from device to pinned host memory.
		cudaMemcpyAsync((h_C + i + segSize * 0), d_C0, stream0Size * sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync((h_C + i + segSize * 1), d_C1, stream1Size * sizeof(float), cudaMemcpyDeviceToHost, stream1);
		cudaMemcpyAsync((h_C + i + segSize * 2), d_C2, stream2Size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
		cudaMemcpyAsync((h_C + i + segSize * 3), d_C3, stream3Size * sizeof(float), cudaMemcpyDeviceToHost, stream3);
	}
	
	// Wait for final iteration's results to be copied from device.
	cudaDeviceSynchronize();
	
	memcpy(hostOutput, h_C, size);
	
	//printf("\noutput[512] = %f = %f", h_C[512], hostOutput[512]);
	
	wbTime_stop(GPU, "Computing results.");
	
	wbTime_start(GPU, "Freeing GPU memory.");
	
	cudaFree(d_A0);
	cudaFree(d_B0);
	cudaFree(d_C0);
	
	cudaFree(d_A1);
	cudaFree(d_B1);
	cudaFree(d_C1);
	
	cudaFree(d_A2);
	cudaFree(d_B2);
	cudaFree(d_C2);
	
	cudaFree(d_A3);
	cudaFree(d_B3);
	cudaFree(d_C3);
	
	wbTime_stop(GPU, "Freeing GPU memory.");
	
	wbTime_start(GPU, "Freeing pinned host memory.");
	
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	
	wbTime_stop(GPU, "Freeing pinned host memory.");
	
    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}