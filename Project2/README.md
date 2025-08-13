# 基于 DCT 的数字水印嵌入与提取

## 1. 项目简介

&emsp;&emsp;本项目实现了基于离散余弦变换（DCT）的数字水印嵌入与提取，并对水印的鲁棒性进行了测试。通过在图像的 DCT 系数中嵌入水印信息，可以在图像经过各种攻击后仍能提取出水印，从而检测图像是否被泄露或篡改。

## 2. 算法数学推导

### 2.1 DCT变换

&emsp;&emsp;离散余弦变换（DCT）是一种将图像从空间域转换到频率域的变换方法。对于一个 8×8 的图像块 \( f(x, y) \)，其 DCT 变换定义为：

$$ 
F(u, v) = C(u)C(v) \sum_{x=0}^{7} \sum_{y=0}^{7} f(x, y) \cos\left(\frac{(2x+1)u\pi}{16}\right) \cos\left(\frac{(2y+1)v\pi}{16}\right) 
$$

其中，\( C(u) \) 和 \( C(v) \) 是归一化系数：

$$ 
C(u) = 
\begin{cases} 
\frac{1}{\sqrt{8}}, & u = 0 \\
\frac{1}{2}, & u \neq 0 
\end{cases}
$$

### 2.2 水印嵌入

&emsp;&emsp;水印嵌入过程是在 DCT 系数中添加水印信息。选择中频系数（如 \( F(3, 4) \)）进行嵌入，以平衡鲁棒性和不可见性：

$$
F'(u, v) = F(u, v) + \alpha \cdot w
$$

其中，\( \alpha \) 是嵌入强度，\( w \) 是水印比特（0 或 1）。

### 2.3 水印提取

&emsp;&emsp;水印提取过程是从 DCT 系数中提取嵌入的水印信息。通过比较嵌入位置的 DCT 系数值来确定水印比特：

$$
w' = 
\begin{cases} 
1, & F'(u, v) > 0 \\
0, & F'(u, v) \leq 0 
\end{cases}
$$


## 3. 核心代码展示

### 3.1 水印嵌入
```python
def embed(self, host_img, watermark):
    # 转换为YUV色彩空间，仅处理Y通道（亮度）
    yuv = cv2.cvtColor(host_img, cv2.COLOR_BGR2YUV)
    y_channel = np.float32(yuv[:, :, 0])
    
    # 将图像分割成8x8块
    blocks = self.split_blocks(y_channel)
    
    # 确保水印是二值图像
    if watermark.dtype != np.uint8 or watermark.max() > 1:
        watermark = (watermark > 128).astype(np.uint8)
    
    # 嵌入水印
    watermarked_blocks = []
    wm_idx = 0
    for block in blocks:
        # 应用DCT变换
        dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
        
        # 在DCT系数中嵌入水印（使用中频）
        if wm_idx < watermark.size:
            dct_block[3, 4] += self.strength * (2 * watermark.flat[wm_idx] - 1)
            wm_idx += 1
        
        # 逆DCT变换
        idct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
        watermarked_blocks.append(idct_block)
    
    # 重构Y通道
    watermarked_y = self.merge_blocks(watermarked_blocks, y_channel.shape)
    
    # 合并通道并转换回BGR
    yuv[:, :, 0] = np.clip(watermarked_y, 0, 255)
    return cv2.cvtColor(np.uint8(yuv), cv2.COLOR_YUV2BGR)
```
### 3.2 水印提取

```python
def extract(self, watermarked_img, watermark_shape):
    yuv = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2YUV)
    y_channel = np.float32(yuv[:, :, 0])
    
    # Split image into 8x8 blocks
    blocks = self.split_blocks(y_channel)
    
    # Extract watermark
    watermark = np.zeros(watermark_shape, dtype=np.uint8)
    wm_idx = 0
    
    for block in blocks:
        if wm_idx >= watermark.size:
            break
            
        # Apply DCT transform
        dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
        
        # Extract watermark bit
        watermark.flat[wm_idx] = 1 if dct_block[3, 4] > 0 else 0
        wm_idx += 1
    
    return watermark * 255  # Convert to 0-255 range
```

## 3. 实验结果分析

### 3.1 鲁棒性测试
#### 整体表现
- 原始水印相似度达 96.75%，说明在无攻击情况下，水印嵌入和提取功能正常，核心逻辑有效。
- 不同攻击下的鲁棒性测试结果差异明显，这符合 DCT 水印的特性（中频嵌入在不可见性和鲁棒性之间取得平衡）。

#### 各类攻击的结果分析

| 攻击类型         | 相似度   | 原因分析                                                                 |
|------------------|----------|--------------------------------------------------------------------------|
| 对比度调整       | 95.02%   | DCT 系数对对比度变化较稳健，对比度调整主要影响整体强度，不破坏频率关系。   |
| 亮度调整         | 95.92%   | 亮度偏移（加法变化）对 DCT 中频系数影响极小，水印信息保留完整。             |
| 高斯噪声         | 96.80%   | 低方差噪声（var=0.02）主要影响高频分量，水印所在的中频区域受干扰较小。   |
| JPEG 压缩        | 96.02%   | JPEG 本质是 DCT 量化，质量 80 时保留了大部分中频系数，水印得以保存。       |
| 平移             | 73.61%   | 平移改变了 8x8 块的位置，部分水印比特丢失，但重叠区域仍保留足够信息。       |
| 旋转 15°         | 54.54%   | 旋转严重破坏块结构，插值过程扭曲了大量 DCT 系数，导致水印信息丢失较多。     |
| 缩放 80%         | 49.78%   | 缩放重采样改变了块边界和系数，比平移更具破坏性，水印完整性受损明显。       |
| 裁剪 20%         | 48.61%   | 直接移除 20% 图像区域，导致大量携带水印的块被丢弃，信息损失显著。           |

#### 结论
- 水印对全局亮度/对比度调整和轻度噪声/压缩表现优异，符合 DCT 算法的设计特性（中频分量抗此类干扰能力强）。
- 对几何变换（旋转、缩放、裁剪）较敏感，这是块基 DCT 水印的固有局限（块结构被破坏后难以恢复）。
- 整体结果与基础 DCT 水印系统的典型性能指标一致，说明代码实现合理，无明显异常。

&emsp;&emsp;综上，输出结果符合理论预期，反映了该水印系统的真实性能特点。

### 3.2 结果可视化
&emsp;&emsp;测试结果已保存为图像文件`robustness_test_results.jpg`，展示了每种攻击后的图像和提取的水印相似度，结果如下：
![SM4-GCM优化性能](https://raw.githubusercontent.com/gml111/Innovation-and-Entrepreneurship-Course-Experiment/main/Project2/results/robustness_test_results.jpg)

## 4. 实验总结
&emsp;&emsp;本项目成功实现了基于DCT的数字水印嵌入与提取，并对水印的鲁棒性进行了测试。实验结果表明，嵌入的水印在经过多种常见图像处理操作后仍能被有效提取，具有较高的鲁棒性。然而，某些攻击（如裁剪和高斯噪声）对水印的提取效果有一定影响，未来可以进一步优化算法以提高水印在这些情况下的鲁棒性。

