# Google Password Checkup 协议实现报告

## 1. 协议数学原理

### 1.1 协议概述
Google Password Checkup 协议允许用户(P1)安全地检查其密码是否出现在泄露数据库(P2)中，而不泄露任何额外信息。协议基于以下密码学原语：
- **双线性映射**：用于隐藏密码标识符
- **同态加密**：用于安全计算交集属性之和
- **哈希函数**：用于密码标识符的匿名化处理

### 1.2 协议数学表示

**参与方**：
- P1：客户端，拥有密码标识符集合
  $V = \\{v_1, v_2, \dots, v_n\\}$
- P2：服务器，拥有泄露密码集合
  $W = \\{(w_1, t_1), (w_2, t_2), \dots, (w_m, t_m)\\}$，其中 $t_j$ 是关联属性（如泄露次数）

**协议流程**：
1. **初始化**：
   - P1 生成随机密钥
     $k_1 \in \mathbb{Z}_p^*$
   - P2 生成随机密钥
     $k_2 \in \mathbb{Z}_p^*$ 和 Paillier 密钥对 $(pk, sk)$

1. **Round 1 (P1 → P2)**  
   - For each $v_i \in V$, compute:  
     $$ Q_i = H(v_i)^{k_1} $$  
   - P1 sends $\left\\{ Q_i \right\\}_{i=1}^n$ **(randomly permuted)** to P2.

---

2. **Round 2 (P2 → P1)**  
   - **Process received items:**  
     For each $Q_i$:  
     $$ Z_i = Q_i^{k_2} = H(v_i)^{k_1k_2} $$  
     
   - **Process own set $W$:**  
     For each $(w_j, t_j) \in W$:  
     $$ R_j = H(w_j)^{k_2}, \quad \text{enc}(t_j) = \text{Enc}_{pk}(t_j) $$  
     
   - P2 sends $Z = \left\\{ Z_i \right\\}$ and $\left\\{ (R_j, \text{enc}(t_j)) \right\\}$ **(randomly permuted)** to P1.

---

3. **Round 3 (P1 → P2)**  
   - For each received $(R_j, \text{enc}(t_j))$:  
     $$ T_j = R_j^{k_1} = H(w_j)^{k_1k_2} $$  
     If $T_j \in Z$, keep $\text{enc}(t_j)$.  
     
   - **Compute encrypted sum:**  
     $$ C = \left( \prod_{\substack{j \in \text{intersection}}} \text{enc}(t_j) \right) \cdot \text{Enc}_{pk}(0) $$  
     
   - P1 sends $C$ to P2.

---

4. **Result Decryption (P2)**  
   $$ \text{sum} = \text{Dec}_{sk}(C) $$

### 1.3 安全属性
- **隐私保护**：P1 不知道 $W$ 的具体内容，P2 不知道 $V$ 的具体内容
- **单向性**：只有交集信息被泄露（属性之和）
- **前向安全**：短期密钥 $k_1, k_2$ 可定期更换

## 2. 主要代码思路

### 2.1 密码学组件实现
- **同态加密系统**：
  - 实现 Paillier 加密方案
  - 支持加密、解密和同态加法
  - 使用 1024 位大素数确保安全性
- **密码标识符处理**：
  - 使用 SHA-256 哈希函数
  - 通过密钥进行伪随机变换模拟指数运算
  - 确保标识符不可逆且不可关联

### 2.2 客户端(P1)实现
- **初始化**：
  - 接收密码标识符集合
  - 生成随机密钥 $k_1$
- **Round 1**：
  - 对每个标识符计算 $H(v_i)^{k_1}$
  - 随机排列结果并发送给服务器
- **Round 3**：
  - 计算服务器标识符的 $H(w_j)^{k_1k_2}$
  - 检测与 $Z$ 集合的交集
  - 对交集元素的属性进行同态求和
  - 添加随机噪声保护交集大小
  - 发送加密结果给服务器

### 2.3 服务器(P2)实现
- **初始化**：
  - 接收泄露密码数据库
  - 生成随机密钥 $k_2$
  - 生成 Paillier 密钥对
- **Round 2**：
  - 计算客户端标识符的 $H(v_i)^{k_1k_2}$ ($Z$ 集合)
  - 对自身标识符计算 $H(w_j)^{k_2}$ 并加密属性
  - 随机排列结果并发送给客户端
- **结果解密**：
  - 解密客户端发回的密文
  - 得到交集属性之和

## 3. 实验结果

### 3.1 测试数据
- **客户端数据集**：`["id1", "id2", "id3", "id5"]`
- **服务器数据集**：`[("id1", 10), ("id2", 20), ("id3", 30), ("id4", 40)]`

### 3.2 预期结果
- **交集元素**：`id1`, `id2`, `id3`
- **交集属性之和**：$10 + 20 + 30 = 60$

### 3.3 实际输出
