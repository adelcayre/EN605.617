#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <time.h>

//size of data;
const int N=12 ;

//add
__global__ void Add(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
                
    Arr_C[i] = Arr_A[i] + Arr_B[i];
                
    
}

//subtract
__global__ void Subtract(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    Arr_C[i] = Arr_A[i] - Arr_B[i];
    
} 

//mult
__global__ void Mult(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    Arr_C[i] = Arr_A[i] * Arr_B[i];
    
} 

//mod
__global__ void Mod(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    Arr_C[i] = Arr_A[i] % Arr_B[i];
    
} 

//branching in kernel
__global__ void Conditional(int* Arr_A, int* Arr_B, int* Arr_C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    //even idx are A mod B, odd idx are A+B
    if(Arr_A[i]%2==0){
        Arr_C[i] = Arr_A[i] % Arr_B[i];;
    }
    if(Arr_A[i]%2==1){
        Arr_C[i] = Arr_A[i] + Arr_B[i];
    }
    
} 

//prepped data
__global__ void ConditionalPrepped(int* Arr_A_o, int* Arr_B_o, int* Arr_C_o, int* Arr_A_e, int* Arr_B_e, int* Arr_C_e)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    //prepped odd arrays are added
    Arr_C_o[i] = Arr_A_o[i] + Arr_B_o[i];;
    
    //prepped even arrays are A mod B
    Arr_C_e[i] = Arr_A_e[i] % Arr_B_e[i];
    
    
} 

void pinnedMemory(char op[], int blocks, int threads, int blockSize){
     
    //declare GPU pointers
    int *dev_a, *dev_b, *dev_c;
    
    //HOST memory allocation
    a = (int*)malloc(N*sizeof(int));
    b = (int*)malloc(N*sizeof(int));
    c = (int*)malloc(N*sizeof(int));
    
    //GPU memory allocation
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    
    //populate host arrays
    srand ( time(NULL) );
    for (int i = 0; i < N; i++) {
        a[i] = i; //a contains index number
        b[i] = rand() % 4; //b contains randoom values 0 to 3
    }
    
    //copy host arrays to GPU
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    
                          
    //performace measurement
    clock_t t;
    
                          
    //select kernel to launch
    if(strcmp(op, "add")==0){
        t=clock();
        Add<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(op, "subtract")==0){
        t=clock();
        Subtract<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(op, "multiply")==0){
        t=clock();
        Mult<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(op, "mod")==0){
        t=clock();
        Mod<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    
    
    
    //copy result back to host
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    //print result for small N
    
    if(N<50){
        for (int i = 0; i < N; i++) {
            printf("%d %d %d \n", a[i], b[i], c[i]);
        }
    }
    
    
    
    double elapsedTime = ((double)t)/CLOCKS_PER_SEC;
    printf("Processed %d %s operations with %d threads and %d blocks (%d threads per block) in %f seconds \n", N, argv[1], threads, blocks, blockSize, elapsedTime);
   
    
    //free allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    
    
}
    
void pageableMemory(char op[], int blocks, int threads, int blockSize){
     
    //declare GPU pointers
    int *dev_a, *dev_b, *dev_c;
    
    //HOST memory allocation
    a = (int*)malloc(N*sizeof(int));
    b = (int*)malloc(N*sizeof(int));
    c = (int*)malloc(N*sizeof(int));
    
    //GPU memory allocation
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    
    //populate host arrays
    srand ( time(NULL) );
    for (int i = 0; i < N; i++) {
        a[i] = i; //a contains index number
        b[i] = rand() % 4; //b contains randoom values 0 to 3
    }
    
    //copy host arrays to GPU
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    
                          
    //performace measurement
    clock_t t;
    
                          
    //select kernel to launch
    if(strcmp(op, "add")==0){
        t=clock();
        Add<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(op, "subtract")==0){
        t=clock();
        Subtract<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(op, "multiply")==0){
        t=clock();
        Mult<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(op, "mod")==0){
        t=clock();
        Mod<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    
    
    
    //copy result back to host
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    //print result for small N
    
    if(N<50){
        for (int i = 0; i < N; i++) {
            printf("%d %d %d \n", a[i], b[i], c[i]);
        }
    }
    
    
    
    double elapsedTime = ((double)t)/CLOCKS_PER_SEC;
    printf("Processed %d %s operations with %d threads and %d blocks (%d threads per block) in %f seconds \n", N, argv[1], threads, blocks, blockSize, elapsedTime);
   
    
    //free allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    
    
}
    


int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[2]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[3]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
    
   
        
        
        
    
    return 0;
    
	}
    
    
    
    
    

}
