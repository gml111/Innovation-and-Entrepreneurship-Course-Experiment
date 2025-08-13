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

## 3. 核心代码实现


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
