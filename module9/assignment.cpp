#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

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
    
    
    //copy result back to host
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    //print result for small N
    
    
    for (int i = 0; i < 20; i++) {
        printf("%d %d %d \n", a[i], b[i], c[i]);
        }
    
    
    
    double elapsedTime = ((double)t)/CLOCKS_PER_SEC;
    printf("Processed %d %s operations with %d threads and %d blocks (%d threads per block) in %f seconds \n", N, argv[1], threads, blocks, blockSize, elapsedTime);
   
    
    //free allocated memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    
    return 0;
    
    }
