#include <cuda.h>
#include <iostream>
#include <vector>
using namespace std;

// Add A and B vector on the GPU. Results stored into C
__global__
void addKernel(int n, float* A, float* B, float* C)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i < n)
        C[i] = A[i] + B[i];
}

// Add A and B vector. Results stored into C
int add(int n, float* h_A, float* h_B, float* h_C)
{
    int size = n*sizeof(float);

    // Allocate memory on device and copy data
    float* d_A;
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    float* d_B;
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    float* d_C;
    cudaMalloc((void**)&d_C, size);

    // launch Kernel
    cout << "Running 256 threads on " << ceil(n/256.0f) << " blocks -> " << 256*ceil(n/256.0f) << endl;
    addKernel<<<ceil(n/256.0f),256>>>(n, d_A, d_B, d_C);

    // Transfer results back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

#define A_VAL   1.0f
#define B_VAL   2.0f
#define C_VAL   3.0f // A_VAL + B_VAL
/**
 * Perform addition operation on 2 vectors A and B using GPU
 * Then verify result
 */
int main(int argc, char* argv[])
{
    int n;
  
    // Check if there are enough command-line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <vector size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    n = atoi(argv[1]);

    vector<float> h_A(n, 1.0f);
    vector<float> h_B(n, 2.0f);
    vector<float> h_C(n);

    add(n, h_A.data(), h_B.data(), h_C.data());

    for(int i = 0; i < h_C.size(); ++i) {
        if(fabs(h_C[i]-3.0f) > 0.00001f) {
            cout << "Validation Failure! C[" << i << "]: " << h_C[i] << endl;
            return EXIT_FAILURE;
        }
    }

    cout << "The program completed successfully" << endl;

    return 0;
}