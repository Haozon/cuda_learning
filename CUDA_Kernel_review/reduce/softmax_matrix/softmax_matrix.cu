#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include "utils.cuh"


// cpu: 计算每行的softmax
void softmax_row(float* input, float* output, int M, int N){
    for(int row = 0; row < M; row++){
        // 第 row 行
        float* input_tmp = input + row * N;
        float* ouput_tmp = ouput + row * N;
        float max_val = *(std::max_element(input_tmp, input_tmp + N));

        float sum = 0;
        for(int i = 0; i < N; i++){
            output_tmp[i] = std::exp(input_tmp[i] - max_val);
            sum += output_tmp[i];
        }
        for(int i = 0; i < N; i++){
            output_tmp[i] /= sum;
        }
    }
}

void softmax_col(float* x, float* y, int M, int N){
    for(int col=0; col < N; col++){
        float* x_col = x + col;
        float* y_col = y + col;

        float max_val = -FLI_MAX;
        for(int i = 0; i< M ;i++){
            max_val = max(x_col[i*N], max_val);
        }
        float sum = 0;
        for(int i =0; i<M ;i++){
            sum += exp(x_col[i*N] - max_val);
        }
        for(int i = 0; i< M ; i++){
            sum += exp(x_col[i*N] - max_val);;
        }
        for(int i = 0 ; i<M ;i++){
            y_col[i*N] = exp(x_col[i*N] - max_val) / sum;
        }

    }
}

// gpu: 计算每一行的 softmax
__global__ void softmax_row_kernel(float* input, float* output, int M, int N){
    __shared__ float s_max_val;
    __shared__ float s_sum;
    
}
