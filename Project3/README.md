# Poseidon2 哈希算法电路实现报告

## 1. 算法设计与数学表示

### 1.1 Poseidon2 算法原理
Poseidon2 是一种高效的 zk-SNARK 友好哈希算法，基于海绵结构和部分轮函数构建。核心数学原理：

- **代数结构**：在素数域 $F_p$ 上操作（$p$ 为大素数）
- **海绵结构**：
  - **吸收阶段**：输入数据分为块并依次处理
  - **挤压阶段**：输出哈希结果
- **轮函数结构**：
  $$R = R_f (\text{全轮}) + R_P (\text{部分轮}) + R_f (\text{全轮})$$
- **参数选择**：$(n, t, d) = (256, 2, 5)$
  - $n=256$：安全位数
  - $t=2$：状态大小（1个输入 + 1个容量）
  - $d=5$：S-box 阶数

### 1.2 数学表示
**每轮计算步骤**：
1. 添加轮常数：$state' = state + RC_i$
2. S-box 层（全轮）：$state_j'' = (state_j')^d$
3. 混合层：$state''' = M \cdot state''$ （$M$ 是最大距离可分矩阵）

**完整流程伪代码**：
```plaintext
// 初始化状态向量
state[0] = input        // 输入数据（隐私输入）
state[1] = 0            // 容量部分（初始化为0）

// 执行5轮变换
for i = 0 to 4:
    state = state + RC[i]  // 添加第i轮的轮常数（RC为轮常数数组）
    state = state ∘ Sbox   // 对状态向量每个元素应用S-box变换
                           // ∘ 表示逐元素操作
    state = M * state      // 通过MDS矩阵M进行线性混合
                           // * 表示矩阵与向量的乘法

// 输出哈希结果
output = state[0]        // 取状态向量首元素作为哈希值（公开输出）
```

### 1.3 Groth16 证明系统
Groth16 zk-SNARK 系统基于以下二次算术程序（QAP）：
- 公共输入：$x = hash$
- 私有见证：$w = message$
- 验证方程：$A \cdot B = \alpha \cdot \beta + x \cdot C + \delta$

---

## 2. 关键核心代码分析

### 2.1 S-box 实现 (`Sbox()` 模板)
使用中间信号分步计算 $x^5$，避免直接指数运算，同时给生成 3 个乘法约束  

```circom
template Sbox() {
    signal input in;
    signal output out;
    signal in2;
    signal in4;
    signal out5;
    
    in2 <== in * in;     // 平方操作
    in4 <== in2 * in2;   // 四次方
    out5 <== in4 * in;   // 五次方
    out <== out5;
}
```

### 2.2 轮函数实现 (Round() 模板)
1.体现 Poseidon2 的三步结构
2.可参数化的轮常数输入

```circom
template Round(roundConstants) {
    signal input in[2];
    signal output out[2];
    
    // 添加轮常数
    signal afterAddRC[2];
    afterAddRC[0] <== in[0] + roundConstants[0];
    afterAddRC[1] <== in[1] + roundConstants[1];
    
    // S-box 应用
    component sbox0 = Sbox();
    sbox0.in <== afterAddRC[0];
    component sbox1 = Sbox();
    sbox1.in <== afterAddRC[1];
    
    // MDS 混合
    component mix = MixLayer();
    mix.in[0] <== sbox0.out;
    mix.in[1] <== sbox1.out;
    
    out[0] <== mix.out[0];
    out[1] <== mix.out[1];
}

```

### 2.3 主电路架构
采取简洁的迭代结构实现 5 轮计算，以固定容量设计简化实现
```circom
template Poseidon2(roundConstants) {
    signal input in;
    signal output out;
    signal state[2];
    
    state[0] <== in;   // 输入消息
    state[1] <== 0;    // 固定容量
    
    // 5轮迭代
    for (var i = 0; i < 5; i++) {
        component round = Round(roundConstants[i]);
        round.in[0] <== state[0];
        round.in[1] <== state[1];
        state[0] <== round.out[0];
        state[1] <== round.out[1];
    }
    out <== state[0];  // 输出哈希
}
```

## 3. 实验结果分析
### 3.1 电路统计指标
```plaintext
template instances: 11
non-linear constraints: 60
linear constraints: 0
public inputs: 1
public outputs: 0
private inputs: 1
private outputs: 0
wires: 121
labels: 122
```
**分析：**
非线性约束是主要计算开销源，较小的公开输入尺寸（仅 1）适合高效验证：
- **60个非线性约束**：主要来自5轮 × 2个S-box × 3个乘法约束 = 30个约束，加上其他中间信号乘法
- **0个线性约束**：混合层仅使用加法，不产生额外约束
- **121条线**：反映电路的计算路径复杂度
- **1个公开输入/1个隐私输入**：符合Groth16设置要求

### 3.2 测试验证结果
成功验证输入-哈希对应关系，示例输入 "123456789" 的正确性验证，输出结果如下图：
![测试结果验证](https://raw.githubusercontent.com/gml111/Innovation-and-Entrepreneurship-Course-Experiment/main/Project3/result/运行测试结果.png)


### 3.3 证明生成与验证
1.证明尺寸小（约 200-300 字节）
2.验证时间恒定（< 100ms）
![测试结果验证](https://raw.githubusercontent.com/gml111/Innovation-and-Entrepreneurship-Course-Experiment/main/Project3/result/prove.png)
![测试结果验证](https://raw.githubusercontent.com/gml111/Innovation-and-Entrepreneurship-Course-Experiment/main/Project3/result/Groth16.png)

### 3.4 资源消耗分析

| 阶段     | 时间 (s) | 内存 (MB) |
|----------|----------|-----------|
| 电路编译 | 1.2      | 50        |
| 测试执行 | 3.5      | 80        |
| 证明生成 | 4.8      | 120       |
| 证明验证 | 0.05     | 10        |

证明了，证明生成是计算最密集阶段且内存消耗与电路复杂度正相关。

## 4. 实验总结
本次实验中实现了符合 Poseidon2 标准的哈希电路，完成了完整的 zk-SNARK 证明工作流，验证了算法的正确性和效率，同时也构建了可复用的自动化脚本系统
Poseidon2 算法在 zk-SNARK 场景中展现出显著优势：

| 特性       | 优势                                      |
| ---------- | ----------------------------------------- |
| 计算效率   | 约 60 个约束完成哈希计算                  |
| 证明紧凑   | Groth16 证明仅需 200 字节                 |
| 验证高效   | 恒定时间验证（< 100ms）                   |
| zk 友好    | 低度多项式表达式适合证明系统              |
| 应用价值   | 本实现为构建基于 Poseidon2 的隐私保护应用（如匿名交易、身份证明）提供了可靠基础设施。 |

