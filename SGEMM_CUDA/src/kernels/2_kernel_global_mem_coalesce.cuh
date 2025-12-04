#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(const float* A,const float* B, float* C, int M, int N, int K, int alpha, int beta){
    const int x = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
    const int y = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

    if(x < M && y < N){
        float temp = 0;
        for(int i = 0; i < K; i++){
            temp += A[x*K + i] * B[i*N + y];
        }
        C[x*N + y] = temp * alpha + beta * C[x*N + y];
    }
}

/*
这里是将每个 block 当作 BLOCKSIZE*BLOCKSIZE 的 tile，是为了 “把矩阵按小块划分，块内线程协同工作、提高数据重用和内存访问效率”
关键原因：
1. 内存合并（coalescing）
    - 令 同一 warp （或相邻线程）访问相邻的全局地址，从而提高带宽利用率。
    - 例如 BLOCKSIZE = 16, threadidx = 0-15 会被映射到同一 tile 的同一行、列 0-15，所以访问 C[x*N + y]（y连续）就是连续内存访问，便于合并；
2. 数据重用（cache、shared memory）
    - tile 内的多个线程会在计算不同 C[x,y] 时用到相同的 A 的某一行 或 B 的某一列。把计算限制在 tile 内后，可以把 A/B 的子块一次载入shared memory ，减少全局内存读取次数；
3. 简化并行映射与边界处理
    - 把 block 视作 tile 后，全局坐标 = blockidx * BLOCKSIZE + tile 内偏移 很直观，也方便处理矩阵尺寸不是 BLOCKSZIE整数倍时的边界判断（if(x<M&y<N))
4. 灵活的线程布局
    - 即使把线程组织成 1D （blockDim.x = BLOCKSIZE * BLOCKSIZE），也能通过 threadidx/BLOCKSIZE 和 threadidx%BLOCKSIZE 把它映射成 tile 内的 (row, col)。即使是使用 2D 的 block,
    思路都是：每个 block 负责一个 BLOCKSIZE x BLOCKSIZE 的子矩阵。 

*/