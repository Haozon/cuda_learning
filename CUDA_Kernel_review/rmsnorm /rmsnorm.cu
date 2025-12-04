// RMSNorm(xᵢ) = (xᵢ / sqrt((1/n) × Σⱼ xⱼ²)) × γᵢ
// 公式说明：
// 1. 计算每行的均方：mean(x²) = (1/K) × Σx²
// 2. 计算均方根的倒数：1/RMS = 1/sqrt(mean(x²) + ε)
// 3. 归一化并缩放：y = x × (1/RMS) × γ

#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32

/*
__shfl_xor_sync 是 CUDA 的 warp shuffle 指令，用于在 warp 內的 32 个线程之间直接交换寄存器数据，无需通过 shared memory。
    float __shfl_xor_sync(unsigned mask, float val, int laneMask);
        mask: 参与同步的线程掩码（通常是 0xffffffff 表示全部的32个线程）
        val: 当前线程要交换的值
        laneMask: XOR 掩码，决定和哪个线程交换数据
    
__forceinline__ 是CUDA的编译器指令，强制编译器内联函数，把函数调用强制替换成函数体本身。 

*/

// ============================================================================
// Warp Reduce Sum (32个线程内归约)
// ============================================================================
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll // CUDA 编译器指令，告诉编译器展开循环(loop unrolling)， 把循环体复制多次，减少循环控制开销。
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
/*
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    
    二分规约的算法(具体讲解参考README)
    核心思想：每次让一半的线程与另一半的线程配对相加。
        由于所有的线程都要参与操作，所以这里是 0xffffffff
        由于使用二分法进行操作，mask 是从 16->8->4->2->1(相邻线程配对)
*/


/*


*/


// ============================================================================
// Block Reduce Sum (整个block归约)
// ============================================================================
template <const int NUM_THREADS = 256>  // 设置 block 內的线程数量是 256 
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    /*
        constexpr: 
            是 C++11 引入的关键字，表示编译器常量表达式，让编译器在编译时就计算出结果；
            也就是表示 NUM_WARPS 在编译时就能确定该值，而不需要等到运行时计算。 
    */
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE; // 获得 block 內 warp 的数量
    int warp = threadIdx.x / WARP_SIZE; // 获得当前线程所在的 warp id
    int lane = threadIdx.x % WARP_SIZE; // 获得当前线程所在 warp 的 lane id  
    
    __shared__ float shared[NUM_WARPS]; // 每个 warp 规约的结果放到共享内存中
    
    // Step 1: 每个warp内归约
    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    
    // Step 2: 每个warp的lane 0写入shared memory
    if (lane == 0) {
        shared[warp] = val;
    }
    /*
        __syncthreads() 函数是 CUDA的线程块內同步函数，确保同一个 block 內所有线程都执行到这个点后才继续；
    */
    __syncthreads();  
    
    // Step 3: 第一个warp对所有warp的结果再次归约
    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f; // 将 shared 中的结果分配到一个 warp 中进行执行。
    val = warp_reduce_sum_f32<NUM_WARPS>(val);
    
    return val;
}

// ============================================================================
// RMSNorm Kernel (Naive版本)
// 输入: x [N, K]  N行，每行K个元素
// 输出: y [N, K]
// 参数: g 是scale因子
// 
// Grid:  (N, 1, 1)     - N个block，每个block处理一行
// Block: (K, 1, 1)     - K个线程，每个线程处理一个元素
// ============================================================================
template <const int NUM_THREADS = 256>
__global__ void rms_norm_f32_kernel(float *x, float *y, float g, int N, int K) {
    int tid = threadIdx.x;  // 0..K-1
    int bid = blockIdx.x;   // 0..N-1
    int idx = bid * blockDim.x + threadIdx.x;
    
    const float epsilon = 1e-5f;
    
    __shared__ float s_variance;  // 存储这一行的 1/RMS
    
    // Step 1: 加载数据并计算平方
    float value = (idx < N * K) ? x[idx] : 0.0f;
    float variance = value * value;
    
    // Step 2: Block内归约求和 Σx²
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    
    // Step 3: 线程0计算 1/RMS = rsqrt(mean(x²) + ε)
    if (tid == 0) {
        s_variance = rsqrtf(variance / (float)K + epsilon);
    }
    __syncthreads();
    
    // Step 4: 每个线程归一化并缩放
    if (idx < N * K) {
        y[idx] = value * s_variance * g;
    }
}

// ============================================================================
// Host函数：启动kernel
// ============================================================================
void rms_norm_f32(float *d_x, float *d_y, float g, int N, int K) {
    dim3 block(K);
    dim3 grid(N);
    
    if (K == 256) {
        rms_norm_f32_kernel<256><<<grid, block>>>(d_x, d_y, g, N, K);
    } else if (K == 512) {
        rms_norm_f32_kernel<512><<<grid, block>>>(d_x, d_y, g, N, K);
    } else if (K == 1024) {
        rms_norm_f32_kernel<1024><<<grid, block>>>(d_x, d_y, g, N, K);
    } else {
        printf("Unsupported K=%d, only support 256/512/1024\n", K);
    }
    
    cudaDeviceSynchronize();
}
