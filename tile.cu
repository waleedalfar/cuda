#include <iostream>
#include <cstdio>
#include <cassert>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr_to_check)                          \
    do                                                     \
    {                                                      \
        cudaError_t result = expr_to_check;                \
        if (result != cudaSuccess)                         \
        {                                                  \
            fprintf(stderr,                                \
                    "CUDA Runtime Error: %s:%i:%d = %s\n", \
                    __FILE__,                              \
                    __LINE__,                              \
                    result,                                \
                    cudaGetErrorString(result));           \
        }                                                  \
    } while (0)

__global__ void tileMult(float *A, float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n)
    {
        return;
    }

    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    float sum = 0.0f;

    for (int tile = 0; tile < n / 32; tile++)
    {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + tile * 32 + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * n + col];
        __syncthreads();
        for (int k = 0; k < 32; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * n + col] = sum;
}

void initMat(float *A, int n)
{
    std::srand(std::time({}));
    for (int i = 0; i < n; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

int main()
{
    std::vector<int> matSizes = {256, 512, 1024, 2048, 4096};
    // host
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;

    // device
    float *devA = nullptr;
    float *devB = nullptr;
    float *devC = nullptr;

    // n x n matrix
    // int n = 1024;

    // int matSize = 1024 * 1024;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (const int size : matSizes)
    {
        // n x n matrix
        int matSize = size * size;

        // allocate for cpu
        CUDA_CHECK(cudaMallocHost(&A, matSize * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&B, matSize * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&C, matSize * sizeof(float)));

        initMat(A, matSize);
        initMat(B, matSize);

        // allocate for gpu
        CUDA_CHECK(cudaMalloc(&devA, matSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&devB, matSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&devC, matSize * sizeof(float)));

        // copy data to the GPU
        CUDA_CHECK(cudaMemcpy(devA, A, matSize * sizeof(float), cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(devB, B, matSize * sizeof(float), cudaMemcpyDefault));
        CUDA_CHECK(cudaMemset(devC, 0, matSize * sizeof(float)));

        dim3 threads(32, 32);
        dim3 blocks(size / 32, size / 32);

        // warmup
        tileMult<<<blocks, threads, 0, stream>>>(devA, devB, devC, size);
        cudaDeviceSynchronize();

        // reset
        cudaMemset(devC, 0, matSize * sizeof(float));

        cudaEventRecord(start, stream);

        tileMult<<<blocks, threads, 0, stream>>>(devA, devB, devC, size);
        cudaEventRecord(stop, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Size: " << size << "x" << size << "\n";
        std::cout << "Kernel execution time: " << elapsedTime << " ms\n";

        float timeInSeconds = elapsedTime / 1000.0f;
        float gflops = (2.0 * size * size * size) / (timeInSeconds * 1e9f);
        std::cout << "GFLOPS: " << gflops << "\n\n";

        cudaFreeHost(A);
        cudaFreeHost(B);
        cudaFreeHost(C);

        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}