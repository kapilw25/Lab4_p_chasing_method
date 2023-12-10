#include <stdio.h>
#include <cuda_runtime.h>

// GPU Kernel
__global__ void P_chasing2(volatile int *A, long long int iterations, long long int *d_tvalue, int stride) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    long long int start_time, end_time;

    // Use a volatile pointer to prevent compiler optimizations
    volatile int *ptr = &A[index];

    for (long long int i = 0; i < iterations; ++i) {
        start_time = clock64();
        index = ptr[index]; // Pointer chasing to generate memory accesses
        ptr[index] = index; // Write to prevent any compiler optimizations
        end_time = clock64();

        // Calculate and write the time taken for each access
        d_tvalue[i] = (end_time - start_time);
    }
}

// Host function to initialize A with a specific stride
void init_cpu_data(int* A, int size, int stride) {
    for (int i = 0; i < size; i++) {
        // Every 'stride' elements are a multiple of the cache line size apart
        A[i] = ((i + stride) % size) * stride;
    }
}

int main() {
    const long long int iterations = 1024;
    const int size = 4096;
    int *h_A = (int *)malloc(size * sizeof(int));
    long long int *h_tvalue = (long long int *)malloc(iterations * sizeof(long long int));

    volatile int *d_A;
    long long int *d_tvalue;
    cudaMalloc((void **)&d_A, size * sizeof(int));
    cudaMalloc((void **)&d_tvalue, iterations * sizeof(long long int));

    // Set the cache preference and output appropriate message
    cudaFuncCache cacheConfigs[2] = {cudaFuncCachePreferL1, cudaFuncCachePreferShared};
    const char *configNames[2] = {"L1 Enabled", "L1 Disabled"};

    for (int config = 0; config < 2; ++config) {
        cudaDeviceSetCacheConfig(cacheConfigs[config]);
        printf("CacheConfig,%s\n", configNames[config]);  // Indicate cache configuration

        for (int stride = 1; stride <= 128; stride++) {  // More granular stride values
            init_cpu_data(h_A, size, stride);
            cudaMemcpy((void *)d_A, h_A, size * sizeof(int), cudaMemcpyHostToDevice);

            P_chasing2<<<1, 1>>>((int *)d_A, iterations, d_tvalue, stride);

            cudaMemcpy(h_tvalue, d_tvalue, iterations * sizeof(long long int), cudaMemcpyDeviceToHost);

            for (int i = 0; i < iterations; ++i) {
                printf("Stride,%d,Iteration,%d,Time,%lld\n", stride, i, h_tvalue[i]);  // Detailed output
            }
        }
    }

    // Cleanup
    cudaFree((void *)d_A);
    cudaFree(d_tvalue);
    free(h_A);
    free(h_tvalue);

    return 0;
}
