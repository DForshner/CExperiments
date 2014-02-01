#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
	
	// Determine current thread's output element row/column	
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	
	// Allocate shared memory tile
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
	
	float value = 0;
	
	// Ceiling function to get sufficent number of tiles needed to cover all the elements of A's x dimension
	// and B's y dimension.
	int xyTiles = (numAColumns - 1) / TILE_WIDTH + 1;
	
	for (int t = 0; t < xyTiles; ++t) {
		int globalAColIndex = t * TILE_WIDTH + threadIdx.x;
		int globalBRowIndex = t * TILE_WIDTH + threadIdx.y;
		
		// Copy all the current tile's input elements into the block's shared memory.
		
		// Check if inside boundary of matrixA
		if ((row < numARows) && (globalAColIndex < numAColumns)) {
			ds_A[threadIdx.y][threadIdx.x] = A[row * numAColumns + globalAColIndex];
		} else {
			ds_A[threadIdx.y][threadIdx.x] = 0;
		}
		
		// Check if inside boundary of matrix B
		if ((col < numBColumns) && (globalBRowIndex < numBRows)) {
			ds_B[threadIdx.y][threadIdx.x] = B[globalBRowIndex * numBColumns + col];
		} else {
			ds_B[threadIdx.y][threadIdx.x] = 0;
		}
		
		// Wait for all threads to finish loading their current tile into shared memory
		__syncthreads();
	
		// Calculate the current threads output element.
		for (int ele = 0; ele < TILE_WIDTH; ++ele) {
			value += ds_A[threadIdx.y][ele] * ds_B[ele][threadIdx.x];
		}
		
		// Wait for all threads to complete calculating output element sub-total for the current tile.
		__syncthreads();
	}
	
	// Check if current row/col is inside the output matrix's boundary
	if ((row < numCRows) && (col < numCColumns)) {
		// Copy result to global output
		C[(row * numCColumns) + col] = value;
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
	
    // Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
	
	// Allocating the hostC matrix in host memory.
	int matrixCSize = numCRows * numCColumns * sizeof(float);
	wbLog(TRACE, "Size of matrix C is ", matrixCSize);
	hostC = (float*) malloc(matrixCSize);
	
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
	
   // Allocate GPU memory
	int matrixASize = numARows * numAColumns * sizeof(float);
	wbLog(TRACE, "Size of matrix A is ", matrixASize);
	wbCheck(cudaMalloc((void **) &deviceA, matrixASize));

	int matrixBSize = numBRows * numBColumns * sizeof(float);
	wbLog(TRACE, "Size of matrix B is ", matrixBSize);
	wbCheck(cudaMalloc((void **) &deviceB, matrixBSize));
	
	wbCheck(cudaMalloc((void **) &deviceC, matrixCSize));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
	
    // Copy memory to the GPU
	wbCheck(cudaMemcpy(deviceA, hostA, matrixASize, cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceB, hostB, matrixBSize, cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    // Initialize the grid and block dimensions
	
	// Use ceiling function to get sufficient # of x tiles.
	int xTiles = ((numCColumns - 1) / TILE_WIDTH + 1);
	wbLog(TRACE, "Number of tiles in x dimension is ", xTiles);
	
	// Use ceiling function to get sufficient # of y tiles.
	int yTiles = ((numCRows - 1) / TILE_WIDTH + 1);
	wbLog(TRACE, "Number of tiles in y dimension is ", yTiles);
	
	dim3 gridDim(xTiles, yTiles, 1);
	wbLog(TRACE, "gridDIM is ", xTiles, ",", yTiles, ",", 1);
	
	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    wbLog(TRACE, "blockDim is ", TILE_WIDTH, ",", TILE_WIDTH, ",", 1);
	
    wbTime_start(Compute, "Performing CUDA computation");

	matrixMultiplyShared<<<gridDim,blockDim>>> (deviceA, deviceB, deviceC,
												numARows, numAColumns,
												numBRows, numBColumns,
												numCRows, numCColumns);
    cudaThreadSynchronize();
	
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
	
    //Copy the GPU memory back to the CPU
	wbCheck(cudaMemcpy(hostC, deviceC, matrixCSize, cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");

    // Free the GPU memory
	wbCheck(cudaFree(deviceA));
	wbCheck(cudaFree(deviceB));
	wbCheck(cudaFree(deviceC));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}