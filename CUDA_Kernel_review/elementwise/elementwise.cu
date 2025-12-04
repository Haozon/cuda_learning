// From https://github.com/Tongkaio/CUDA_Kernel_Samples/blob/master/elementwise/add.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

#define CEIL(a, b) ((a+b-1)/b)


__global__ void elementwise_add_f32_kernel(float* a, float* b, float*c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_add_f32x4_kernel(float* a, float* b, float* c, int N){
    int idx =  4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx < N){
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(a[idx]);
        flaot4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_c.z + reg_c.z;
        reg_c.w = reg_c.w + reg_c.w;
        FLOAT4(c[idx]) = reg_c;
    }
}

int main(){
    constexpr int N = 7;
    float* a_h = (float*)malloc(N * sizeof(float));
    float* b_h = (float*)malloc(N * sizeof(float));
    float* c_h = (float*)malloc(N * sizeof(float));
    for(int i = 0; i < N; i++){
        a_h[i] = i;
        b_h[i] = N - 1 - i;

    }
    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;
    
    int block_size = 1024;
    int grid_size = CEIL(CEIL(N,4), 1024);

    elementwise_add_f32x4_kernel<<<grid_size, block_size>>>(a_d, b_d, c_d, N);

    for(int i = 0; i < N; i++){
        if( i == N -1) printf("%f\n", a_h[i]);
        else printf("%f ", a_h[i]);
    }
    
}

 