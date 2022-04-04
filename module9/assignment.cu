#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>
#include <time.h>       


const int N=20;

int main(int argc, char* argv[]) {
    
    //declare host vectors
    thrust::host_vector<int> A(N);
    thrust::host_vector<int> B(N);
    thrust::host_vector<int> C(N);
    
    //populate vectors
    thrust::sequence(A.begin(), A.end()); //A elements contain their index
    for (int i = 0; i < N; i++) {
        B[i] = rand() % 4; //B contains randoom values 0 to 3
    }
    
    //performace measurement
    clock_t elapsedTime;
    elapsedTime= clock();

    //declare device vectors/copy to device 
    thrust::device_vector<int> A_dev(A);
    thrust::device_vector<int> B_dev(B);
    thrust::device_vector<int> C_dev(N);
                              
    //select kernel to launch
    if(strcmp(argv[1], "add")==0){
      thrust::transform(A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<int>());

    }
    
    else if(strcmp(argv[1], "subtract")==0){
        
    }
    
    else if(strcmp(argv[1], "multiply")==0){
        
    }
    
    else if(strcmp(argv[1], "mod")==0){
    }
   
    //copy result back to host
    thrust::copy(C_dev.begin(), C_dev.end(), C.begin());
    elapsedTime= clock()-elapsedTime;
    
    
    
    //print result
    for(int i = 0; i < 20; i++){
        std::cout << A[i] << " " << B[i] << " " << C[i] << std::endl;
    }
    std::cout << "Using cuda thrust it took "  << elapsedTime << " seconds to " << argv[1] << " two vectors size "<< N << std::endl;

    
    return 0;
    
    
} 
