#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include "utils.cuh"


const int N = 2048;
constexpr size_t BLOCK_SIZE = 256;
const int repeat_times = 10;

__global__ void setToNegativeMax(float* d_value){
    *d_value = -FLT_MAX;
}

__device__ static float atomicMax(float* address, float val){
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do{
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    }while(assumed != old);
    return __int_as_float(old);
}

__global__ void max_kernel(float* float, float* output, int N){
    __shared__ float s_mem[32]; 
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int LaneId = threadIdx.x % warpSize;
    
    float val = (idx < N)? input[x] : (-FLT_MAX) // FLT_MAX 即 float 能表示的最大数值 
    /*
        __shfl_down_sync 是 warp shuffle 指令，可以让当前线程从 "offset" 距离下一个线程那里胡哦的 val 的值。
    */
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    } // warp 内规约

    if(LaneId == 0)s_mem[swarpId] = val; // 每个 warp 的第一个线程存储的是 warp 内规约后的数值，将该数值存到共享内存中进行 warp 间的数据交互。

    __syncthreads(); // 整个线程块内的线程进行同步

    if(warpId == 0){// 第一个 warp 负责将在 s_mem 中所有 warp 计算得到的数值进行 warp 内规约。
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum)? s_mem[laneId] : (-FLT_MAX);
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if(laneId == 0) atomicMax(output, val);
    }
}


__global__ void softmax_kernel(float* input, float* output, float* sum, float* max_val, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) output[idx] = expf(input[idx] - *max_val) / (*sum);
}

void softmax_kernel(float* input, float* output, int N, flkoat* M, float* sum){
    *M = *(std::max_element(input, input+N));
    *sum = 0;
    for(int i = 0; i< N;i ++){
        output[i] = std::exp(input[i] - *M);
        *sum += output[i];
    }
    for(int i = 0; i < N; i++){
        output[i] /= *sum;
    }
}

void call_softmax_v1(float* output, float* input_device, float* output_device, float* total_device, float* total_max_device, int N){
    int block_size = BLOCK_SIZE;
    int grid_size = CEIL(N, BLOCK_SIZE);

    // 1 初始化
    cudaCheck(cudaMemset(total_device, 0, sizeof(float)));
    cudaCheck(cudaMemset(total_max_device, 0, sizeof(float)));

    // 2 计算和
    sum_kernel<<<grid_size, block_size>>>(input_device, output_device, total_device, total_max_device, N);

    // 3 计算和
    sum_kernel<<<grid_size, block_size>>>(input_device, total_device, total_max_device, N);

    // 4 计算 softmax (减去最大值避免溢出)
    softmax_kernel<<<grid_size, block_size>>>(input_device, output_device, total_device, total_max_device, N);
}

int main(){
    float* input = (float*)malloc(sizeof(float)*N);
    float* output_ref = (float*)malloc(sizeof(float)*N);
    float* M = (float*)malloc(sizeof(float));
    float* sum = (float*)malloc(sizeof(float));
    for(int i = 0; i< N; i++){
        input[i] = i / (float)N;
    }
    float total_time_h = TIME_RECORD(repeat_times, ([&]{softmax(input, output_ref, N, M, sum)}));

    float* input_device = nullptr;
    float* output_device = nullptr;
    float* total_device = nullptr;
    float* total_max_device = nullptr;
    cudaCheck(cudaMalloc(&input_device, N * sizeof(float)));
    cudaCheck(cudaMalloc(&output_device, N * sizeof(float)));
    cudaCheck(cudaMalloc(&total_device, 1 * sizeof(float)));
    cudaCheck(cudaMalloc(&total_max_device, 1 * sizeof(float)));

    cudaCheck(cudaMemcpy(input_device, input, N * sizeof(float), cudaMemcpyHostToDevice));
    float* output = (float*)malloc(sizeof(float) * N);

    // softmax_v1
    float total_time_1 = TIME_RECORD(repeat_times, ([&]{call_softmax_v1(output, input_device, output_device, total_device, total_max_device, N);}));
    printf("[softmax_kernel1]: total_time_1 = %f ms\n", total_time_1 / repeat_times);
    cudaCheck(cudaMemcpy(output, output_device, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); 
    verify_matrix(output, output_ref, N);

    // softmax_v2
    float total_time_2 = TIME_RECORD(repeat_times, ([&]{call_softmax_v2(output, input_device, output_device, total_device, total_max_device, N);}));
    printf("[softmax_kernel2]: total_time_2 = %f ms\n", total_time_2 / repeat_times);
    cudaCheck(cudaMemcpy(output, output_device, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    verify_matrix(output, output_ref, N);

    float* total_host = (float*)malloc(sizeof(float));
    float* total_max_host = (float*)malloc(sizeof(float));
    cudaCheck(cudaMemcpy(total_host, total_device, sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(total_max_host, total_max_device, sizeof(float), cudaMemcpyDeviceToHost));

    free(input);
    free(output);
    free(M);
    free(sum);
    free(output_ref);
    cudaCheck(cudaFree(input_device));
    cudaCheck(cudaFree(output_device));
    cudaCheck(cudaFree(total_device));
    cudaCheck(cudaFree(total_max_device));
return 0;
}
