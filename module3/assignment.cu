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
    



int main(int argc, char* argv[]) {
	// read command line arguments
	int threads = atoi(argv[2]);
	int blockSize = atoi(argv[3]);
    int blocks= threads/blockSize;
	
    
    
    //declare host arrays and GPU pointers
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    
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
    if(strcmp(argv[1], "add")==0){
        t=clock();
        Add<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(argv[1], "subtract")==0){
        t=clock();
        Subtract<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(argv[1], "multiply")==0){
        t=clock();
        Mult<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(argv[1], "mod")==0){
        t=clock();
        Mod<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
    }
    
    else if(strcmp(argv[1], "conditional")==0){
        
        
        //splitting a and b into odd and even arrays by index
        int a_odd[N/2], a_even[N/2], b_odd[N/2], b_even[N/2];
        int *dev_a_odd,*dev_a_even, *dev_b_odd, *dev_b_even, *dev_c_odd, *dev_c_even;
        
        //allocate space for new arrays
        cudaMalloc((void**)&dev_a_even, N/2 * sizeof(int));
        cudaMalloc((void**)&dev_b_even, N/2 * sizeof(int));
        cudaMalloc((void**)&dev_c_even, N/2 * sizeof(int));
        cudaMalloc((void**)&dev_a_odd, N/2 * sizeof(int));
        cudaMalloc((void**)&dev_b_odd, N/2 * sizeof(int));
        cudaMalloc((void**)&dev_c_odd, N/2 * sizeof(int));
    
        //time for prepared data starts when data prep begins
        
        
        //split odd and even indexes into separate arrays
        for(int i=0; i<N; i++) {
            if(i%2==0){
                a_even[i/2]=a[i];
                b_even[i/2]=b[i];
                          }
            else if(i%2==1){
                a_odd[i/2]=a[i];
                b_odd[i/2]=b[i];
                          }
            }
        //copy even host arrays to GPU
        cudaMemcpy(dev_a_even, a_even, N/2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b_even, b_even, N/2 * sizeof(int), cudaMemcpyHostToDevice);  
        cudaMemcpy(dev_a_odd, a_odd, N/2 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b_odd, b_odd, N/2 * sizeof(int), cudaMemcpyHostToDevice);                    
        
        //execute single kernel with prepped data
        t=clock();
        ConditionalPrepped<<<blocks, threads>>>(dev_a_odd, dev_b_odd, dev_c_odd, dev_a_even, dev_b_even, dev_c_even);
        t=clock()-t;
        
        double preppedTime = ((double)t)/CLOCKS_PER_SEC;
        printf("With data prep this process took %f seconds \n", preppedTime);
    
        //execute single kernel with branching
        t=clock();
        Conditional<<<blocks, threads>>>(dev_a, dev_b, dev_c);
        t=clock()-t;
        
        cudaFree(dev_a_odd);
        cudaFree(dev_b_odd);
        cudaFree(dev_c_odd);
        cudaFree(dev_a_even);
        cudaFree(dev_b_even);
        cudaFree(dev_c_even);
        }
    
    
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
    
    
    return 0;
    
	}
    
    
    
    
    
