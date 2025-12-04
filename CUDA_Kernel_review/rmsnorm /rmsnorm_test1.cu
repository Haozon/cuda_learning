#define WARP_SIZE 32

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(int val){
    #pragma unroll
    for(int mask = kWarpSize << 1 ; mask >= 1; mask <<= 1){
        __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}



template<const int THREADS_NUM = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val){
    constexpr int WAPR_NUM = (THREADS_NUM + WAPR_SIZE - 1)  / (WARP_SIZE);

    int warp = threadIdx.x / WAPR_SIZE;
    int lane = threadIdx.x % WAPR_SIZE;

    // 定义共享内存
    __shared__ float shared[NUM_WARPS]; // 共享内存大小 是 warps 的数量

    // 该线程执行 warp 內规约
    val = warp_reduce_sum_f32<WAPR_SIZE>(val);

    // 只有每个 warp 的 lane 0 才写入共享内存
    if(lane == 0)shared[warp] = val;

    // 同步
    __syncthreads();

    // 让一个 warp 对 shared 中的数据 进行规约
    val = (lane < WAPR_NUM)? shared[lane]:0.0f;
    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    return val ;
}


/*
x: M * K 
y: M * K
w: K 
*/

template<const int NUM_THREADS  = 256>
__global__ float rms_norm_f32_kernel(float *x, float *y, float* w, int M, int K){
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int idx = bid * blockDim.x + tid;

    const float epslion = 1e-5f;

    __shared__ float s_variance; // 存储这一行的 1/RMS 

    // 1. 加载数据并计算平方
    float value = (idx < M * K)? x[idx] : 0.0f;
    float variance = value * value;

    // 2. block 內规约求和
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);

    // 3. 线程 0 计算 1 / RMS  = rsqrt(mean(x^2) + epslion)
    if(tid == 0){
        s_variance = rsqrtf(variance / (float)K +_ epsilon);
    }
    __syncthreads();
    // 4. 每个线程归一化并缩放
    float g = (tid < K)? w[tid]: 1.0f;
    if(idx < N * K){
        y[idx] = value * s_variance * g;
    }
}
