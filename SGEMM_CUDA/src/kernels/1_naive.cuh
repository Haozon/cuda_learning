#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(const float* A,const float* B, float* C, int M, int N, int K, float alpha, float beta){
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < M && y < N){
        float tmp = 0.0;
        for(int i = 0; i < K; i++){
            tmp += A[x * K + i] * B[i * M + y];
        }
        // C = a * (A@B) + beat * C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }       
}