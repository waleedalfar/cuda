#include <iostream>
#include <cstdio>
#include <cassert>
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

__global__ void matMult(float *A, float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n)
    {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n; k++)
    {
        sum += A[row * n + k] * B[k * n + col];
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
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;

    int n = 1024;

    CUDA_CHECK(cudaMallocManaged(&A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C, n * n * sizeof(float)));

    initMat(A, n * n);
    initMat(B, n * n);

    dim3 threads(32, 32);
    dim3 blocks(n / 32, n / 32);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    matMult<<<blocks, threads, 0, stream>>>(A, B, C, n);
    cudaEventRecord(stop, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Kernel execution time: " << elapsedTime << " ms\n";

    float timeInSeconds = elapsedTime / 1000.0f;
    float gflops = (2.0 * n * n * n) / (timeInSeconds * 1e9f);
    std::cout << "GFLOPS: " << gflops << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}