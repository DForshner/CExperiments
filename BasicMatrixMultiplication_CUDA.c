#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
                   int numARows, int numAColumns,
                   int numBRows, int numBColumns,
                   int numCRows, int numCColumns) {

	// Determine current result matrix element to calculate.
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	
	//C[x,y] = DotProduct(A.row(x), B.col(y))
	
	// Check if current row/col is inside result matrix boundary
	if ((row < numCRows) && (col < numCColumns)) {
		
		// Calculate inner product for the current A row and B column.
		float value = 0;
		for (int i = 0; i < numAColumns; ++i) {
			// Linearize locations of input matrix a/b elements.
			int aElement = row * numAColumns + i;
			int bElement = i * numBColumns + col;
			
			value += A[aElement] * B[bElement];
		}
		
		// Store element in result matrix
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
	int xTiles = ((numCColumns - 1) / TILE_WIDTH + 1);
	wbLog(TRACE, "Number of tiles in x dimension is ", xTiles);
	int yTiles = ((numCRows - 1) / TILE_WIDTH + 1);
	wbLog(TRACE, "Number of tiles in y dimension is ", yTiles);
	
	dim3 gridDim(xTiles, yTiles, 1);
	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");
	
	// Check A&B matrix dimensions
	if (numAColumns != numBRows)
		wbLog(ERROR, "Invalid input matrix dimensions");
	
    // Launch the GPU Kernel
	matrixMultiply<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, 
										  numBRows, numBColumns, numCRows, numCColumns);
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

	// Free host memory
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}