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
    float sum = 0.0f;
    for (int k = 0; k < n; k++)
    {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
    printf("%f\n", sum);
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

    int n = 32;

    CUDA_CHECK(cudaMallocManaged(&A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C, n * n * sizeof(float)));

    initMat(A, n * n);
    initMat(B, n * n);

    dim3 threads(n, n);
    dim3 blocks(n / 32, n / 32);

    matMult<<<blocks, threads>>>(A, B, C, 32);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}