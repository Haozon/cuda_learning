// From https://github.com/Tongkaio/CUDA_Kernel_Samples/tree/master/gemv
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 

#define CEIL(a,b) ((a)+((b)-1))/(b)
#define checkCudaErrors(func) {                                                   \
    cudaError_t e = (func);                                                       \
    if(e != cudaSuccess)                                                          \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));  \
}


// 适合K∈[32,128]使用，小于32或大于128有进一步的优化方法
__global__ void sgemv_k32(float* A, float* x, float* y, int M, int K){
    int laneId = threadIdx.x % warpSize;
    int row = blockIdx.x; // 0 ~ M - 1
    if(row >= M ) return;

    float res = 0.0f;
    int kIteration = CEIL(k, warpSize ); // 每个线程需要负责计算的数据个数

    #pragma unroll
    for(int i = 0; i < kIteration; i++){
        int col = i * warpSize + laneId;
        res += (col < K) ? A[row * K + col] * x[col] : 0.0f;

    }
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        res += __shfl_down_sync(0xFFFFFFFF, res, offset);
    }

    if(laneId == 0) y[row] =res;

}

int main(){
    size_t M = 1024;
    size_t K = 32;

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_x = sizeof(float) * K;
    size_t bytes_y = sizeof(float) * M;
    float* h_A = (float*)malloc(bytes_A);
    float* h_x = (float*)malloc(bytes_x);
    float* h_y = (float*)malloc(bytes_y);
    float* h_y1 = (float*)malloc(bytes_y);

    float* d_A;
    float* d_x;
    float* d_y;

    double duration[2] = {0,0};
    double GFLOPS[2] = {0,0};
    double GFLOPS = 2.0 * M * 1 * K;

    for(int i = 0; i< M * K ;i++){
        h_A[i] = (float)i/k;
    }

    for(int i = 0; i< K ; i++){
        h_x[i] = 1;
    }
    memset(h_y, 0, M*sizeof(float));
    memset(h_y1, 0, M*sizeof(float));

    float mesecTotal = 0;
    int iteration = 1000;

    for(int run = 0; run < iteration; run++){
        dim3 dimGrid(M);
        dim3 dimBlock(32);
        sgemv_k32<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, K);
    }

    duration[0] = mesecTotal / iteration;
    GFLOPS[0] = (GFLOPS * 1.0e-9f) / (duration[0] / 1000.0f);
    

}