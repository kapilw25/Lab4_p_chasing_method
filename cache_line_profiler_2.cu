
// To enhance the analysis of the timing data for identifying patterns or changes in access times that correlate with cache line boundaries,
//  you can modify the code to include more granular data collection and analysis. Here's an updated version of your code with these modifications:

// Key Modifications:
// 1) Granular Stride Iteration: The stride now iterates more granularly, starting from 1 up to 128. This finer granularity helps 
// in identifying the cache line size more accurately.
// 2) Detailed Timing Analysis: Instead of just printing individual access times, the code now calculates the average access time 
// for each stride. A sudden increase in average time could indicate crossing a cache line boundary.
// 3) Threshold for Detecting Changes: A simple heuristic is used to detect significant changes in average access times. If the 
// average time for a particular stride is substantially higher than the previous stride (e.g., more than 20% higher), 
// it's flagged as a potential cache line boundary

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
        printf("Profiling with %s...\n", configNames[config]);

        // Iterate over a range of strides to cover possible cache line sizes
        for (int stride = 1; stride <= 128; stride++) {
            init_cpu_data(h_A, size, stride);
            cudaMemcpy((void *)d_A, h_A, size * sizeof(int), cudaMemcpyHostToDevice);

            P_chasing2<<<1, 1>>>((int *)d_A, iterations, d_tvalue, stride);

            // Copy the timing values back to the host
            cudaMemcpy(h_tvalue, d_tvalue, iterations * sizeof(long long int), cudaMemcpyDeviceToHost);

            // Analyzing and outputting the timing data
            long long int sum = 0;
            for (int i = 0; i < iterations; ++i) {
                sum += h_tvalue[i];
            }
            long long int avg = sum / iterations;
            printf("Average time for stride %d: %lld cycles\n", stride, avg);

            // Detect sudden changes in average time, which may indicate cache line boundary
            if (stride > 1) {
                long long int prev_avg = sum / (iterations * (stride - 1));
                if (avg > prev_avg * 1.2) { // Threshold for change, can be adjusted
                    printf("Possible cache line boundary detected at stride %d\n", stride);
                }
            }
        }

        printf("Completed profiling with %s\n", configNames[config]);
    }

    // Cleanup
    cudaFree((void *)d_A);
    cudaFree(d_tvalue);
    free(h_A);
    free(h_tvalue);

    return 0;
}
