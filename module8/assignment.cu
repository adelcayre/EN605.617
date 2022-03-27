#include <stdio.h>
#include <cublas.h>

const int N= 20;
const int MAX= 100;

//utility function prints arrays
void printVector(int * vec)
{   
    printf("{");
    for(int x = 0 ; x < N ; x++){
            printf(", %f", vec[x]);
        }
    printf("}\n");
}



void vectorSub(){
    //declare GPU pointers
    int *a, *b, *c, *dev_a, *dev_b, *dev_c;
    
    //HOST pinned memory allocation
    cudaMallocHost((int **)&a, N*sizeof(int));
    cudaMallocHost((int **)&b, N*sizeof(int));
    cudaMallocHost((int **)&c, N*sizeof(int));
    
    //GPU memory allocation
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    
    //populate host arrays
    srand ( time(NULL) );
    for (int i = 0; i < N; i++) {
        a[i] = rand() % MAX; //a rand between 0-99
        b[i] = rand() % MAX; //b rand between 0-99
    }
    
    //performace measurement
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start,0);
    cudaEventCreate(&kernel_stop,0);
    
    
    //initiallize
    cudaEventRecord(kernel_start, 0);
    cublasInit();
    
    //set vector on device
    cublasSetVector(N, sizeof(int), a, 1, dev_a, 1);
    cublasSetVector(N, sizeof(int), b, 1, dev_b, 1);
       
    //saxpy with a=-1 to subtract a from b
    cublasSaxpy(N, -1.0, dev_a, 1, dev_b, 1);
    
    //copy back and shutdown cublas
    cublasGetVector(N, sizeof(int), dev_c, 1, c, 1);
    cublasShutdown();
    cudaEventRecord(kernel_stop, 0);
    
    
    //output results and time
    float elapsedTime =0.0F;
    cudaEventElapsedTime(&elapsedTime, kernel_start, kernel_stop);
    printf("Processed saxpy operations on vectors size %d using cublas in %f seconds using pinned memmory \n", N, elapsedTime);
    printVector(a);
    printVector(b);
    printVector(c);
    
    
    
    //free allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    
    
    
    
}

int main(int argc, char** argv)
{
    vectorSub();
   
}
