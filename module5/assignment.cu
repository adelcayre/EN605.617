#include <stdio.h>

//timing
__host__ cudaEvent_t get_time(void)
{
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}


//add const
__global__ void add_const(int* Arr_C)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N) Arr_C[i] = dev_a[i] + dev_b[i];
                
    
}

__global__ void add(int* Arr_C){
    
}

//subtract
__global__ void subtract(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N) Arr_C[i] = Arr_A[i] - Arr_B[i];
    
} 

//mult
__global__ void mult(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N) Arr_C[i] = Arr_A[i] * Arr_B[i];
    
} 

//mod
__global__ void mod(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N) Arr_C[i] = Arr_A[i] % Arr_B[i];
    
}
    

__host__ void launchKernelsConst(char op[], int blocks, int threads, int blockSize){
    
    //declare pointers
    int *a, *b, *c,
    
    //declare gpu constant memory arrays
    __const__  int dev_a[N], dev_b[N]; 
    
    //HOST pinned memory allocation
    cudaMallocHost((void **)&a, N*sizeof(int));
    cudaMallocHost((void **)&b, N*sizeof(int));
    cudaMallocHost((void **)&c, N*sizeof(int));
    
    
    //GPU memory allocation for result
    cudaMalloc(&dev_c, N * sizeof(int));
    
    //direct constant memory allocation
    __device__ __constant__ int dev_a_const[N]
    
    //populate host arrays and constant memory allocation
    srand ( time(NULL) );
    for (int i = 0; i < N; i++) {
        a[i] = i; //a contains index number
        b[i] = rand() % 4; //b contains randoom values 0 to 3
    }
    
    //copy to GPU const memory    
    cudaMemcpyToSymbol(dev_a_const, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_b_const, b, N * sizeof(int), cudaMemcpyHostToDevice);
    
    
    //performace measurement
    cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);
       
                          
    //select kernel to launch 
    if(strcmp(op, "add")==0){
        cudaEventRecord(kernel_start, 0);
        add<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        cudaEventRecord(kernel_stop, 0);    }
    
    else if(strcmp(op, "subtract")==0){
        cudaEventRecord(kernel_start, 0);
        subtract<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        cudaEventRecord(kernel_stop, 0);
    }
    
    else if(strcmp(op, "multiply")==0){
        cudaEventRecord(kernel_start, 0);
        mult<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        cudaEventRecord(kernel_stop, 0);
    }
    
    else if(strcmp(op, "mod")==0){
        cudaEventRecord(kernel_start, 0);
        mod<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        cudaEventRecord(kernel_stop, 0);
    }
    
    //synchronize 
    cudaEventSynchronize(kernel_stop);

    //copy result back to host
    cudaMemcpyFromSymbol(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    //print first 10 elements
    for (int i = 0; i < 10; i++) {
        printf("%d %d %d \n", a[i], b[i], c[i]);
    }

    
      
    float elapsedTime =0.0F;
    cudaEventElapsedTime(&elapsedTime, start_time, end_time);
    printf("Processed %d %s operations with %d threads and %d blocks (%d threads per block) in %f seconds using constant memory \n", N, op, threads, blocks, blockSize, elapsedTime);
   
    //free allocated memory
    free(a);
    free(b);
    free(c);
    
    //destroy events
    cudaEventDestroy(kernel_start1);
    cudaEventDestroy(kernel_start2);
    
}

int main(int argc, char** argv)
{
    int main(int argc, char** argv){
	// read command line arguments
    
	int totalThreads = 1024;
	int blockSize = 256;
	
	if (argc >= 3) {
		totalThreads = atoi(argv[2]);
	}
	if (argc >= 4) {
		blockSize = atoi(argv[3]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
    
   
        }
        
    launchKernelsConst(argv[1], numBlocks, totalThreads, blockSize);
	
}
