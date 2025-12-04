#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_global_mem_block(const float* A, const float* B, float* C, int M, int N, int K, int alpha, int beta){
    // the output block that we want to compute in this threadblock    
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE]
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE]

    // the inner row & col that we're accessing in this thread
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K; // row = cRow, col = 0
    B += cCol * BLOCKSIZE; // row = 0, col = cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row = cRow, col = col

    float tmp = 0.0;
    for(int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE){
        // Have each thread load one of elements in A&B
        // Make the threadCol (=threadId.x) the consecutive index
        // to allow global memory access coalescing 
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // excute the dotproduct  on the currently cache block
        for(int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx){
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // need to syncthreads again at the end, to aviod faster threads
        __syncthreads();

    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}


__global__ void sgemm_global_mem_block(const float* A, const float* B, const float* C, int M, int N, int K , int alpha, int beta){
    // 获得对应线程块
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;
    
    // 分配共享内存
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // 获得该线程执行的对应内积的索引
    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;

    // 移动 A，B，C的 ptr 用于获取对应的数据
    A += cRow * BLOCKSIZE * K; // row = cRow, col = 0
    B += cCol * BLOCKSIZE;  // row = 0, col = cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; //  row = cRow, col = cCol

    float tmp = 0.0;
    for(int bkIdx= 0; bkIdx < K; bkIdx += BLOCKSIZE){
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();

        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for(int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx){
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}