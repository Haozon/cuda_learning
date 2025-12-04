# RMSNorm CUDA 实现详解

## 目录
- [RMSNorm 原理](#rmsnorm-原理)
- [Warp Shuffle 归约详解](#warp-shuffle-归约详解)
- [代码实现](#代码实现)

---

## RMSNorm 原理

### 公式
```
RMSNorm(xᵢ) = (xᵢ / sqrt((1/n) × Σⱼ xⱼ²)) × γᵢ
```

### 计算步骤
1. **计算每行的均方**：`mean(x²) = (1/K) × Σx²`
2. **计算均方根的倒数**：`1/RMS = 1/sqrt(mean(x²) + ε)`
3. **归一化并缩放**：`y = x × (1/RMS) × γ`

---

## Warp Shuffle 归约详解

### `__shfl_xor_sync` 函数

#### 函数原型
```cpp
float __shfl_xor_sync(unsigned mask, float val, int laneMask);
```

**参数说明：**
- `mask`: 参与同步的线程掩码（通常用 `0xffffffff` 表示全部32个线程）
- `val`: 当前线程要交换的值
- `laneMask`: XOR 掩码，决定和哪个线程交换数据

#### 工作原理
每个线程会和 `threadIdx.x ^ laneMask` 位置的线程交换数据。

---

### laneMask 深度理解

#### laneMask 的本质

**laneMask 是一个位掩码，用于计算"我要和哪个线程交换数据"**

核心公式：
```cpp
目标线程ID = 当前线程ID XOR laneMask
```

#### 二进制视角

每个线程在 warp 内有一个 lane ID（0-31），用 5 位二进制表示：

```
Lane ID    二进制表示
   0       00000
   1       00001
   2       00010
   3       00011
   ...
   16      10000
   17      10001
   ...
   31      11111
```

#### laneMask = 16 的例子

```
laneMask = 16 = 10000 (二进制)
                ↑
                第4位是1，其他位是0
```

**XOR 操作会"翻转"第 4 位：**

```
线程 0:  00000 XOR 10000 = 10000 (16)  ← 和线程16交换
线程 1:  00001 XOR 10000 = 10001 (17)  ← 和线程17交换
线程 2:  00010 XOR 10000 = 10010 (18)  ← 和线程18交换
...
线程 15: 01111 XOR 10000 = 11111 (31)  ← 和线程31交换
线程 16: 10000 XOR 10000 = 00000 (0)   ← 和线程0交换
线程 17: 10001 XOR 10000 = 00001 (1)   ← 和线程1交换
```

**规律：前 16 个线程和后 16 个线程配对！**

#### laneMask = 8 的例子

```
laneMask = 8 = 01000 (二进制)
               ↑
               第3位是1
```

**XOR 操作翻转第 3 位：**

```
线程 0:  00000 XOR 01000 = 01000 (8)   ← 和线程8交换
线程 1:  00001 XOR 01000 = 01001 (9)   ← 和线程9交换
...
线程 7:  00111 XOR 01000 = 01111 (15)  ← 和线程15交换
线程 8:  01000 XOR 01000 = 00000 (0)   ← 和线程0交换
线程 9:  01001 XOR 01000 = 00001 (1)   ← 和线程1交换
```

**规律：每 8 个线程为一组配对！**

#### laneMask = 1 的例子

```
laneMask = 1 = 00001 (二进制)
                   ↑
                   第0位是1
```

**XOR 操作翻转第 0 位（最低位）：**

```
线程 0:  00000 XOR 00001 = 00001 (1)   ← 和线程1交换
线程 1:  00001 XOR 00001 = 00000 (0)   ← 和线程0交换
线程 2:  00010 XOR 00001 = 00011 (3)   ← 和线程3交换
线程 3:  00011 XOR 00001 = 00010 (2)   ← 和线程2交换
```

**规律：相邻线程配对！**

#### 可视化：laneMask 控制配对距离

```
laneMask 的二进制位决定了配对的"跨度"

laneMask = 16 (10000):  翻转第4位 → 跨度 2^4 = 16
    0 ←─────────────→ 16
    1 ←─────────────→ 17
    ...

laneMask = 8 (01000):   翻转第3位 → 跨度 2^3 = 8
    0 ←───────→ 8
    1 ←───────→ 9
    ...

laneMask = 4 (00100):   翻转第2位 → 跨度 2^2 = 4
    0 ←───→ 4
    1 ←───→ 5
    ...

laneMask = 2 (00010):   翻转第1位 → 跨度 2^1 = 2
    0 ←→ 2
    1 ←→ 3
    ...

laneMask = 1 (00001):   翻转第0位 → 跨度 2^0 = 1
    0 ↔ 1
    2 ↔ 3
    ...
```

#### 为什么用 XOR 而不是加法？

##### 对比：XOR vs 加法

假设想让线程 0 和线程 16 配对：

**方案1：加法（不对称）**
```cpp
target = (threadIdx.x + 16) % 32;
```
- 线程 0  → 16 ✓
- 线程 16 → 0  ✗ (16+16=32, 32%32=0 ✓ 但需要取模)
- 线程 20 → 4  ✓
- **问题：需要额外的取模运算**

**方案2：XOR（对称）**
```cpp
target = threadIdx.x ^ 16;
```
- 线程 0  → 16 ✓
- 线程 16 → 0  ✓ (自动对称！)
- 线程 20 → 4  ✓
- **优势：天然对称，无需取模**

##### XOR 的数学性质

```
A XOR B XOR B = A  (自反性)
A XOR B = B XOR A  (交换律)
```

这意味着：
```cpp
// 线程A和线程B互相交换
A_new = A_old + (B_old)  // A从B获取数据
B_new = B_old + (A_old)  // B从A获取数据

// 用同一个laneMask，两边都能正确配对！
```

#### 完整示例：8 个线程的归约

```cpp
// 假设 warpSize = 8（简化示例）
for (int mask = 4; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xff, val, mask);
}
```

##### 初始状态
```
Lane:  0   1   2   3   4   5   6   7
Val:   1   2   3   4   5   6   7   8
```

##### 第1轮：mask = 4 (100)
```
XOR 配对表：
0 (000) XOR 4 (100) = 4 (100)  →  1+5=6
1 (001) XOR 4 (100) = 5 (101)  →  2+6=8
2 (010) XOR 4 (100) = 6 (110)  →  3+7=10
3 (011) XOR 4 (100) = 7 (111)  →  4+8=12
4 (100) XOR 4 (100) = 0 (000)  →  5+1=6
5 (101) XOR 4 (100) = 1 (001)  →  6+2=8
6 (110) XOR 4 (100) = 2 (010)  →  7+3=10
7 (111) XOR 4 (100) = 3 (011)  →  8+4=12

结果: [6, 8, 10, 12, 6, 8, 10, 12]
```

##### 第2轮：mask = 2 (010)
```
0 (000) XOR 2 (010) = 2 (010)  →  6+10=16
1 (001) XOR 2 (010) = 3 (011)  →  8+12=20
2 (010) XOR 2 (010) = 0 (000)  →  10+6=16
3 (011) XOR 2 (010) = 1 (001)  →  12+8=20
...

结果: [16, 20, 16, 20, 16, 20, 16, 20]
```

##### 第3轮：mask = 1 (001)
```
0 (000) XOR 1 (001) = 1 (001)  →  16+20=36
1 (001) XOR 1 (001) = 0 (000)  →  20+16=36
...

最终: [36, 36, 36, 36, 36, 36, 36, 36]
```

#### laneMask 关键理解

1. **laneMask 是二进制掩码**，每个位控制一个维度的配对
2. **XOR 操作翻转特定位**，实现对称配对
3. **mask >>= 1** 每次减半，对应二分归约的层级
4. **所有线程同时工作**，没有线程空闲
5. **天然对称性**，无需额外的边界检查或取模运算

---

### 核心代码分析

```cpp
#pragma unroll
for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
}
```

这是一个**二分归约**算法，每轮让一半的线程和另一半配对相加。

---

### 逐轮分解（32 个线程示例）

假设 `kWarpSize = 32`，初始每个线程有一个值：

```
线程ID:  0   1   2   3   4  ...  16  17  18  19  ...  31
初始值:  a0  a1  a2  a3  a4 ...  a16 a17 a18 a19 ...  a31
```

#### 第 1 轮：mask = 16

```cpp
mask = 32 >> 1 = 16
val += __shfl_xor_sync(0xffffffff, val, 16);
```

**XOR 配对规则：**
```
线程 0  (00000) XOR 16 (10000) = 线程 16  →  a0  + a16
线程 1  (00001) XOR 16 (10000) = 线程 17  →  a1  + a17
线程 2  (00010) XOR 16 (10000) = 线程 18  →  a2  + a18
...
线程 16 (10000) XOR 16 (10000) = 线程 0   →  a16 + a0
线程 17 (10001) XOR 16 (10000) = 线程 1   →  a17 + a1
```

**图示：**
```
前16个线程          后16个线程
    ↓                   ↓
 0  1  2  3  ...     16 17 18 19 ...
 │  │  │  │          │  │  │  │
 └──┼──┼──┼──────────┘  │  │  │
    └──┼──┼─────────────┘  │  │
       └──┼────────────────┘  │
          └───────────────────┘

结果: 每个线程都有两个数的和
线程 0:  a0+a16
线程 1:  a1+a17
...
线程 31: a31+a15
```

#### 第 2 轮：mask = 8

```cpp
mask >>= 1;  // mask = 8
val += __shfl_xor_sync(0xffffffff, val, 8);
```

**XOR 配对：**
```
线程 0 (00000) XOR 8 (01000) = 线程 8   →  (a0+a16) + (a8+a24)
线程 1 (00001) XOR 8 (01000) = 线程 9   →  (a1+a17) + (a9+a25)
...
```

**图示：**
```
每8个线程一组配对
 0  1  2  3  4  5  6  7     8  9 10 11 12 13 14 15
 │  │  │  │  │  │  │  │     │  │  │  │  │  │  │  │
 └──┼──┼──┼──┼──┼──┼──┼─────┘  │  │  │  │  │  │  │
    └──┼──┼──┼──┼──┼──┼────────┘  │  │  │  │  │  │
       ...

结果: 每个线程有 4 个原始数的和
```

#### 第 3 轮：mask = 4

```cpp
mask >>= 1;  // mask = 4
val += __shfl_xor_sync(0xffffffff, val, 4);
```

```
每4个线程一组配对
 0  1  2  3     4  5  6  7
 │  │  │  │     │  │  │  │
 └──┼──┼──┼─────┘  │  │  │
    └──┼──┼────────┘  │  │
       └──┼───────────┘  │
          └──────────────┘

结果: 每个线程有 8 个原始数的和
```

#### 第 4 轮：mask = 2

```cpp
mask >>= 1;  // mask = 2
val += __shfl_xor_sync(0xffffffff, val, 2);
```

```
每2个线程一组配对
 0  1     2  3
 │  │     │  │
 └──┼─────┘  │
    └────────┘

结果: 每个线程有 16 个原始数的和
```

#### 第 5 轮：mask = 1

```cpp
mask >>= 1;  // mask = 1
val += __shfl_xor_sync(0xffffffff, val, 1);
```

```
相邻线程配对
 0     1
 │     │
 └─────┘

结果: 所有 32 个线程都有完整的和！
```

---

### 完整流程可视化

```
32个线程的归约树（从下往上看）:

轮次    mask    活跃配对距离
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
初始     -      a0  a1  a2  a3  a4  a5  a6  a7 ... a24 a25 ... a31
                │   │   │   │   │   │   │   │      │   │       │
第1轮    16     └───┼───┼───┼───┼───┼───┼───┼──────┘   │       │
                    └───┼───┼───┼───┼───┼───┼──────────┘       │
                        └───┼───┼───┼───┼───┼──────────────────┘
                            ...
                            
第2轮    8      每8个一组配对
                            
第3轮    4      每4个一组配对
                            
第4轮    2      每2个一组配对
                            
第5轮    1      相邻配对
                            
结果     -      所有线程都有 sum(a0...a31)
```

---

### mask 右移的含义

```cpp
mask = 32 >> 1 = 16  // 二进制: 10000
mask >>= 1     = 8   // 二进制: 01000
mask >>= 1     = 4   // 二进制: 00100
mask >>= 1     = 2   // 二进制: 00010
mask >>= 1     = 1   // 二进制: 00001
mask >>= 1     = 0   // 循环结束
```

**每次右移 1 位 = 距离减半**，正好对应二分归约的每一层。

---

### 关键理解

#### 1. XOR 的对称性
`A XOR B == B XOR A`
- 线程 0 和线程 16 互相交换数据
- 两边都能得到对方的值

#### 2. mask 控制配对距离
- `mask=16`: 距离 16 的线程配对
- `mask=8`:  距离 8 的线程配对
- `mask=1`:  相邻线程配对

#### 3. 所有线程都参与
- 不像传统归约只有一半线程工作
- 每个线程都在每轮执行相同操作
- 最后所有线程都有完整结果

---

---

## `#pragma` 关键字详解

### pragma 的读音和含义

#### 读音
**`pragma`** 读作：**[ˈpræɡmə]** (PRAG-muh)
- 重音在第一音节
- 类似 "普拉格玛"

#### 词源
**不是缩写！** 它是一个完整的希腊语单词：

```
pragma (πρᾶγμα)
├─ 希腊语：πρᾶγμα (pragma)
├─ 意思：deed, act, thing done（行为、行动、实际的事）
└─ 词根：πράσσω (prasso) = to do, to act（做、行动）
```

#### 在编程中的含义
**`#pragma`** = **"实用指令"** 或 **"编译器指令"**

它告诉编译器："这是一个**实际操作指令**，不是代码逻辑"

#### 记忆方法

**方法1：联想 "pragmatic"（实用主义的）**
```
pragmatic (实用的、务实的)
    ↓
pragma (实用指令)
    ↓
#pragma unroll (实用的循环展开指令)
```

**记忆口诀：**
> "pragma 很 pragmatic（务实），直接告诉编译器怎么做"

**方法2：对比记忆**

| 关键字 | 含义 | 作用对象 |
|--------|------|---------|
| `if/for/while` | 控制代码逻辑 | 运行时 |
| `#define/#include` | 预处理宏 | 编译前 |
| `#pragma` | **编译器行为指令** | 编译时 |

#### 常见的 pragma 指令

**CUDA 中：**
```cpp
#pragma unroll           // 展开循环
#pragma unroll 4         // 展开4次
#pragma once             // 防止头文件重复包含
```

**C/C++ 中：**
```cpp
#pragma pack(1)          // 结构体字节对齐
#pragma warning(disable: 4996)  // 禁用警告
#pragma omp parallel     // OpenMP 并行化
```

#### 类比理解
```cpp
// 代码逻辑（告诉程序做什么）
for (int i = 0; i < 10; i++) {
    sum += data[i];
}

// pragma 指令（告诉编译器怎么优化）
#pragma unroll
for (int i = 0; i < 10; i++) {
    sum += data[i];
}
```

**类比：**
- 代码 = 菜谱（告诉厨师做什么菜）
- pragma = 烹饪技巧（告诉厨师怎么做更好）

---

## `#pragma unroll` 详解

### 基本概念

`#pragma unroll` 是 CUDA 编译器指令，告诉编译器**展开循环**（loop unrolling），把循环体复制多次，减少循环控制开销。

### 不展开 vs 展开

#### 不展开的循环（原始代码）
```cpp
for (int mask = 16; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
}
```

编译后的伪代码：
```cpp
mask = 16;
loop_start:
    if (mask < 1) goto loop_end;
    val += __shfl_xor_sync(0xffffffff, val, mask);
    mask >>= 1;
    goto loop_start;
loop_end:
```

#### 展开后的循环
```cpp
#pragma unroll
for (int mask = 16; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
}
```

编译后直接展开成：
```cpp
val += __shfl_xor_sync(0xffffffff, val, 16);
val += __shfl_xor_sync(0xffffffff, val, 8);
val += __shfl_xor_sync(0xffffffff, val, 4);
val += __shfl_xor_sync(0xffffffff, val, 2);
val += __shfl_xor_sync(0xffffffff, val, 1);
```

### 优势对比

| 特性 | 不展开 | 展开 |
|------|--------|------|
| **指令数** | 循环控制 + 5次迭代 | 只有5条指令 |
| **分支预测** | 需要判断循环条件 | 无分支 |
| **寄存器压力** | 需要 mask 变量 | mask 是常量 |
| **延迟** | 更高 | 更低 |
| **代码大小** | 小 | 大 |

### 使用场景

#### ✅ 适合展开
```cpp
// 1. 循环次数固定且较小
#pragma unroll
for (int i = 0; i < 8; i++) {
    sum += data[i];
}

// 2. 循环体简单
#pragma unroll
for (int mask = 16; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
}
```

#### ❌ 不适合展开
```cpp
// 1. 循环次数很大（会导致代码膨胀）
#pragma unroll  // 不推荐！
for (int i = 0; i < 10000; i++) {
    sum += data[i];
}

// 2. 循环次数运行时才知道
#pragma unroll  // 无效！
for (int i = 0; i < n; i++) {  // n 是变量
    sum += data[i];
}
```

---

## Warp Reduce 的最终结果

### 关键结论

经过完整的 warp reduce 后：
- ✅ **所有 32 个线程的 `val` 都相同**
- ✅ **每个线程都持有完整的归约结果**
- ✅ **返回值在所有线程中一致**

### 完整过程验证（8线程简化示例）

```cpp
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = 4; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xff, val, mask);
    }
    return val;  // 所有8个线程返回相同的值！
}
```

#### 逐轮演示

```
初始状态（每个线程有不同的值）:
Lane:  0   1   2   3   4   5   6   7
Val:   1   2   3   4   5   6   7   8
       ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓

第1轮 mask=4:
Lane:  0   1   2   3   4   5   6   7
Val:   6   8  10  12   6   8  10  12
       ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓

第2轮 mask=2:
Lane:  0   1   2   3   4   5   6   7
Val:  16  20  16  20  16  20  16  20
       ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓

第3轮 mask=1:
Lane:  0   1   2   3   4   5   6   7
Val:  36  36  36  36  36  36  36  36  ← 所有线程都是36！
       ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
      相同 相同 相同 相同 相同 相同 相同 相同
```

### 为什么所有线程都有相同的值？

#### XOR 的对称性

每一轮，**所有线程都参与计算**，而且是**对称交换**：

```
第1轮 (mask=4):
  线程0 和 线程4 交换  →  两边都得到 (val0 + val4)
  线程1 和 线程5 交换  →  两边都得到 (val1 + val5)
  ...

第2轮 (mask=2):
  线程0 和 线程2 交换  →  两边都得到 (val0 + val2 + val4 + val6)
  线程1 和 线程3 交换  →  两边都得到 (val1 + val3 + val5 + val7)
  ...

第3轮 (mask=1):
  线程0 和 线程1 交换  →  两边都得到 sum(val0...val7)
  线程2 和 线程3 交换  →  两边都得到 sum(val0...val7)
  ...

最终：所有线程都有 sum(val0...val7)
```

### 在 Block Reduce 中的应用

```cpp
template <const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    
    __shared__ float shared[NUM_WARPS];
    
    // Step 1: Warp 内归约
    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    // 此时：每个 warp 内的 32 个线程都有相同的 val
    
    // Step 2: 每个 warp 的 lane 0 写入 shared memory
    if (lane == 0) {
        shared[warp] = val;  // 只需要一个线程写，因为都相同！
    }
    __syncthreads();
    
    // Step 3: 第一个 warp 对所有 warp 的结果再次归约
    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum_f32<NUM_WARPS>(val);
    
    return val;  // 所有线程返回相同的值！
}
```

#### 执行流程说明

```
假设 block 有 256 个线程 = 8 个 warp

Step 1: Warp 内归约
  Warp 0: 线程 0-31   都有 sum(warp0)
  Warp 1: 线程 32-63  都有 sum(warp1)
  ...
  Warp 7: 线程 224-255 都有 sum(warp7)

Step 2: 写入 shared memory
  shared[0] = sum(warp0)  ← 只有 lane 0 写
  shared[1] = sum(warp1)
  ...
  shared[7] = sum(warp7)

Step 3: 第一个 warp 再次归约
  线程 0-7:  加载 shared[0-7]
  线程 8-31: 加载 0.0f
  
  经过 warp_reduce_sum 后：
  线程 0-31: 都有 sum(所有256个线程)

最终结果：
  线程 0-31:   有完整的和
  线程 32-255: 有各自 warp 的和（但通常不使用）
```

### 在 RMSNorm 中的实际应用

```cpp
template <const int NUM_THREADS = 256>
__global__ void rms_norm_f32_kernel(float *x, float *y, float g, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + threadIdx.x;
    
    __shared__ float s_variance;
    
    float value = (idx < N * K) ? x[idx] : 0.0f;
    float variance = value * value;
    
    // 所有线程都参与归约
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    // 此时：所有线程的 variance 都相同！
    
    // 只需要一个线程写入 shared memory
    if (tid == 0) {
        s_variance = rsqrtf(variance / (float)K + epsilon);
    }
    __syncthreads();
    
    // 所有线程读取相同的 s_variance
    if (idx < N * K) {
        y[idx] = value * s_variance * g;
    }
}
```

#### 为什么只用 `tid == 0` 写入？

```
因为 block_reduce_sum 后：
  线程 0:   variance = 100.0
  线程 1:   variance = 100.0
  线程 2:   variance = 100.0
  ...
  线程 255: variance = 100.0

所有线程都有相同的值，所以：
  - 只需要一个线程写入 shared memory
  - 避免重复写入（虽然写入相同值也没问题）
  - 节省带宽
```

### 验证代码示例

```cpp
__global__ void test_warp_reduce() {
    int lane = threadIdx.x % 32;
    float val = (float)(lane + 1);  // 1, 2, 3, ..., 32
    
    // Warp reduce
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    
    // 所有线程都应该有 sum(1..32) = 528
    printf("Lane %d: val = %f (expected 528.0)\n", lane, val);
}
```

**预期输出：**
```
Lane 0: val = 528.0 (expected 528.0)
Lane 1: val = 528.0 (expected 528.0)
Lane 2: val = 528.0 (expected 528.0)
...
Lane 31: val = 528.0 (expected 528.0)
```

### 核心优势

这也是为什么 warp shuffle 比传统归约更高效：

| 特性 | 传统归约 | Shuffle 归约 |
|------|---------|-------------|
| **结果分布** | 只有部分线程有结果 | **所有线程都有结果** |
| **需要广播** | 是（需要额外操作） | 否（天然广播） |
| **同步开销** | 需要 shared memory | 寄存器直通 |
| **延迟** | ~20 cycles | ~1 cycle |

---

## 代码实现

### 三层结构

```
warp_reduce_sum_f32()     ← 32线程用shuffle指令归约
        ↓
block_reduce_sum_f32()    ← 多个warp通过shared memory归约
        ↓
rms_norm_f32_kernel()     ← 主kernel，每个block处理一行
```

### 执行流程

```
线程布局: grid(N), block(K)
├─ 每个block处理矩阵的一行
├─ Step 1: 每个线程加载1个元素并平方
├─ Step 2: block_reduce_sum 计算 Σx²
├─ Step 3: 线程0计算 rsqrt(mean + ε)
└─ Step 4: 每个线程归一化 y = x * (1/RMS) * g
```

### 性能对比

| 方法 | 延迟 | 带宽 | 复杂度 |
|------|------|------|--------|
| **Shuffle** | ~1 cycle | 寄存器直通 | O(log N) |
| Shared Memory | ~20 cycles | 需要同步 | O(log N) |
| Global Memory | ~400 cycles | 极慢 | O(log N) |

---

## 学习建议

1. 先在纸上画出 N=2, K=256 的线程分配图
2. 手动 trace 一遍 warp reduce 的 shuffle 过程
3. 用 nsight compute 对比 naive vs vectorized 的带宽差异
4. 理解为什么需要 `__syncthreads()` 两次

---

## 参考资料

- [CUDA Programming Guide - Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
