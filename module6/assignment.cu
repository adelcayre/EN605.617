#include <stdio.h>
    
//array size
const int N=256;
    
    
 //declare gpu constant memory arrays
 __constant__  int dev_a[N], dev_b[N]; 

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N) Arr_C[i] = dev_a[i] + dev_b[i];        
    
}


//subtract const
__global__ void subtract_const(int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N) Arr_C[i] = dev_a[i] - dev_b[i];
    
} 

//mult const
__global__ void mult_const(int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N) Arr_C[i] = dev_a[i] * dev_b[i];
    
} 

//mod const
__global__ void mod_const(int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N) Arr_C[i] = dev_a[i] % dev_b[i];
    
}


//add shared
__global__ void Add(int* Arr_A, int* Arr_B, int* Arr_C)
{
   
   __shared__ int a[N];
   __shared__ int b[N];
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   a[i]=Arr_A[i];
   b[i]=Arr_B[i];
   __syncthreads();
   
   if(i<N) Arr_C[i] = a[i] + b[i] + 2;
                
    
}

//subtract shared
__global__ void Subtract(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int a[N];
    __shared__ int b[N];
    a[i]=Arr_A[i];
    b[i]=Arr_B[i];
   __syncthreads();
    if(i<N) Arr_C[i] = a[i] - b[i];
    
} 

//mult shared
__global__ void Mult(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int a[N];
    __shared__ int b[N];
    a[i]=Arr_A[i];
    b[i]=Arr_B[i];
   __syncthreads();
    if(i<N) Arr_C[i] = a[i] * b[i];
    
} 

//mod shared
__global__ void Mod(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int a[N];
    __shared__ int b[N];
    a[i]=Arr_A[i];
    b[i]=Arr_B[i];
   __syncthreads();
    if(i<N) Arr_C[i] = a[i] % b[i];
    
} 
    


                           
void allocateShared(){
     
    //declare pointers
    int *a, *b, *c, *dev_a, *dev_b, *dev_c;
    
    //HOST pinned memory allocation
    cudaMallocHost((void **)&a, N*sizeof(int));
    cudaMallocHost((void **)&b, N*sizeof(int));
    cudaMallocHost((void **)&c, N*sizeof(int));
    
    //GPU memory allocation
    cudaMalloc(&dev_a, N * sizeof(int));
    cudaMalloc(&dev_b, N * sizeof(int));
    cudaMalloc(&dev_c, N * sizeof(int));
    
    //populate host arrays
    srand ( time(NULL) );
    for (int i = 0; i < N; i++) {
        a[i] = i; //a contains index number
        b[i] = rand() % 4; //b contains randoom values 0 to 3
    }
    
    //copy host arrays to GPU
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
}

void launchKernelShared(char op[], blocks, threads, blockSize){
                          
    //performace measurement
    cudaEvent_t kernel_start, kernel_stop;
	cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);
                          
    //select kernel to launch
    if(strcmp(op, "add")==0){
        cudaEventRecord(kernel_start, 0);
        Add<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        cudaEventRecord(kernel_stop, 0);
    }
    
    else if(strcmp(op, "subtract")==0){
        cudaEventRecord(kernel_start, 0);
        Subtract<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        cudaEventRecord(kernel_stop, 0);
    }
    
    else if(strcmp(op, "multiply")==0){
        cudaEventRecord(kernel_start, 0);
        Mult<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        cudaEventRecord(kernel_stop, 0);
    }
    
    else if(strcmp(op, "mod")==0){
        cudaEventRecord(kernel_start, 0);
        Mod<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        cudaEventRecord(kernel_stop, 0);
    }
    
    //synchronize 
    cudaEventSynchronize(kernel_stop);
    cudaDeviceSynchronize();
    
    
    //copy result back to host
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    //print first 10 elements of result
    for (int i = 0; i < 10; i++) {
        printf("%d %d %d \n", a[i], b[i], c[i]);
    }
    
    
    float elapsedTime =0.0F;
    cudaEventElapsedTime(&elapsedTime, kernel_start, kernel_stop);
    printf("Processed %d %s operations with %d threads and %d blocks (%d threads per block) in %f seconds using shared memmory \n", N, op, threads, blocks, blockSize, elapsedTime);
}

void freeShared() {   
    //free allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    
    //destroy events
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
}                           

int main(int argc, char** argv)
{
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
    
    allocateShared();    
    launchKernelShared(argv[1], numBlocks, totalThreads, blockSize);
    freeShared();
	
}
