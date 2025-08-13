#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <immintrin.h>
#include <cstdint>

// 使用单个常量矩阵优化SM4 S盒
static const __m128i SM4_GFNI_MATRIX = _mm_set_epi64x(0x040E0606020C0E01, 0x0B0D0501080A0D04);

// SM4 S-Box (保留用于基础实现)
static const uint8_t SBOX[256] = {
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48
};

// 系统参数 (FK)
static const uint32_t FK[4] = {
    0xA3B1BAC6, 0x56AA3350, 0x677D9197, 0xB27022DC
};

// 固定参数 (CK)
static const uint32_t CK[32] = {
    0x00070E15, 0x1C232A31, 0x383F464D, 0x545B6269,
    0x70777E85, 0x8C939AA1, 0xA8AFB6BD, 0xC4CBD2D9,
    0xE0E7EEF5, 0xFC030A11, 0x181F262D, 0x343B4249,
    0x50575E65, 0x6C737A81, 0x888F969D, 0xA4ABB2B9,
    0xC0C7CED5, 0xDCE3EAF1, 0xF8FF060D, 0x141B2229,
    0x30373E45, 0x4C535A61, 0x686F767D, 0x848B9299,
    0xA0A7AEB5, 0xBCC3CAD1, 0xD8DFE6ED, 0xF4FB0209,
    0x10171E25, 0x2C333A41, 0x484F565D, 0x646B7279
};

// 循环左移
inline uint32_t ROTL(uint32_t x, uint8_t n) {
    return (x << n) | (x >> (32 - n));
}

// 优化的GFNI S盒实现
inline uint32_t tau_optimized(uint32_t A) {
    // 将32位整数加载到128位寄存器的低32位
    __m128i input = _mm_cvtsi32_si128(A);
    // 使用单个GFNI指令完成S盒替换
    // 0xD3是SM4特定的GFNI常量，SM4_GFNI_MATRIX是预定义的仿射变换矩阵
    __m128i result = _mm_gf2p8affine_epi64_epi8(input, SM4_GFNI_MATRIX, 0xD3);
    // 正确提取结果寄存器的低32位
    return static_cast<uint32_t>(_mm_cvtsi128_si32(result));
}

// 基础实现的S盒替换
inline uint32_t tau(uint32_t A) {
    uint32_t B = 0;
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A >> 24)]) << 24;
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A >> 16)]) << 16;
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A >> 8)]) << 8;
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A)]);
    return B;
}

// 轮函数的线性变换L
inline uint32_t L(uint32_t B) {
    return B ^ ROTL(B, 2) ^ ROTL(B, 10) ^ ROTL(B, 18) ^ ROTL(B, 24);
}

// 密钥扩展的线性变换L'
inline uint32_t L_prime(uint32_t B) {
    return B ^ ROTL(B, 13) ^ ROTL(B, 23);
}

// 加密用的T函数 - 带参数的版本，用于性能测试
inline uint32_t T(uint32_t X, bool useOptimized) {
    if (useOptimized) {
        return L(tau_optimized(X));
    }
    else {
        return L(tau(X));
    }
}

// 密钥扩展用的T'函数 - 带参数的版本，用于性能测试
inline uint32_t T_prime(uint32_t X, bool useOptimized) {
    if (useOptimized) {
        return L_prime(tau_optimized(X));
    }
    else {
        return L_prime(tau(X));
    }
}

// 密钥扩展 - 带参数的版本，用于性能测试
void SM4_KeySchedule(const uint8_t key[16], uint32_t rk[32], bool forEncryption, bool useOptimized) {
    uint32_t K[36];
    // 初始化中间密钥
    for (int i = 0; i < 4; ++i) {
        K[i] = static_cast<uint32_t>(key[4 * i] << 24) |
            static_cast<uint32_t>(key[4 * i + 1] << 16) |
            static_cast<uint32_t>(key[4 * i + 2] << 8) |
            static_cast<uint32_t>(key[4 * i + 3]);
        K[i] ^= FK[i];
    }
    // 生成轮密钥
    for (int i = 0; i < 32; ++i) {
        const uint32_t X = K[i + 1] ^ K[i + 2] ^ K[i + 3] ^ CK[i];
        K[i + 4] = K[i] ^ T_prime(X, useOptimized);
        // 按正确顺序存储轮密钥（加密或解密）
        if (forEncryption) {
            rk[i] = K[i + 4];
        }
        else {
            rk[31 - i] = K[i + 4];
        }
    }
}

// 处理一个128位块（加密/解密通用）- 带参数的版本，用于性能测试
void SM4_CryptBlock(const uint8_t in[16], uint8_t out[16], const uint32_t rk[32], bool useOptimized) {
    uint32_t X[36];
    // 加载输入字（大端序）
    for (int i = 0; i < 4; ++i) {
        X[i] = static_cast<uint32_t>(in[4 * i] << 24) |
            static_cast<uint32_t>(in[4 * i + 1] << 16) |
            static_cast<uint32_t>(in[4 * i + 2] << 8) |
            static_cast<uint32_t>(in[4 * i + 3]);
    }
    // 32轮SM4
    for (int i = 0; i < 32; ++i) {
        const uint32_t X_next = X[i] ^ T(X[i + 1] ^ X[i + 2] ^ X[i + 3] ^ rk[i], useOptimized);
        X[i + 4] = X_next;
    }
    // 以反转顺序存储输出（大端序）
    for (int i = 0; i < 4; ++i) {
        uint32_t word = X[35 - i];
        out[4 * i] = static_cast<uint8_t>(word >> 24);
        out[4 * i + 1] = static_cast<uint8_t>(word >> 16);
        out[4 * i + 2] = static_cast<uint8_t>(word >> 8);
        out[4 * i + 3] = static_cast<uint8_t>(word);
    }
}

// 性能测试函数
double benchmarkSM4(const uint8_t key[16], const uint8_t plaintext[16],
    uint8_t ciphertext[16], uint8_t decrypted[16],
    bool useOptimized, size_t iterations) {
    uint32_t encrypt_rk[32];
    uint32_t decrypt_rk[32];

    // 生成密钥（计入性能测试，因为密钥扩展也使用了S盒）
    auto start = std::chrono::high_resolution_clock::now();

    SM4_KeySchedule(key, encrypt_rk, true, useOptimized);
    SM4_KeySchedule(key, decrypt_rk, false, useOptimized);

    // 执行多次加密解密操作
    for (size_t i = 0; i < iterations; ++i) {
        SM4_CryptBlock(plaintext, ciphertext, encrypt_rk, useOptimized);
        SM4_CryptBlock(ciphertext, decrypted, decrypt_rk, useOptimized);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 验证结果正确性
    if (memcmp(plaintext, decrypted, 16) != 0) {
        std::cerr << "Error: Decryption mismatch!" << std::endl;
    }

    return elapsed.count();
}

// 辅助函数：打印十六进制输出
void PrintHex(const char* label, const uint8_t* data, size_t len) {
    std::cout << label;
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(data[i]) << " ";
    }
    std::cout << std::dec << std::endl;
}

int main() {
    // 测试参数
    const size_t iterations = 1000000;  // 迭代次数，可根据性能调整

    // 来自GB/T 32907-2016的示例密钥和明文
    uint8_t key[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
    };
    uint8_t plaintext[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
    };
    uint8_t ciphertext[16];
    uint8_t decrypted[16];

    // 显示测试数据
    PrintHex("Key:         ", key, 16);
    PrintHex("Plaintext:   ", plaintext, 16);

    // 测试基础实现性能
    std::cout << "\nRunning benchmark for basic implementation..." << std::endl;
    double basicTime = benchmarkSM4(key, plaintext, ciphertext, decrypted, false, iterations);

    // 显示基础实现的加密结果
    PrintHex("Ciphertext (basic):  ", ciphertext, 16);
    PrintHex("Decrypted (basic):   ", decrypted, 16);

    // 测试GFNI优化实现性能
    std::cout << "\nRunning benchmark for GFNI optimized implementation..." << std::endl;
    double optimizedTime = benchmarkSM4(key, plaintext, ciphertext, decrypted, true, iterations);

    // 显示优化实现的加密结果
    PrintHex("Ciphertext (optimized):  ", ciphertext, 16);
    PrintHex("Decrypted (optimized):   ", decrypted, 16);

    // 计算性能指标
    double basicThroughput = (iterations * 16.0) / (basicTime * 1024 * 1024);  // MB/s
    double optimizedThroughput = (iterations * 16.0) / (optimizedTime * 1024 * 1024);  // MB/s
    double speedup = basicTime / optimizedTime;

    // 显示性能对比结果
    std::cout << "\nPerformance Comparison:" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Basic implementation:   " << basicTime << " seconds for "
        << iterations << " iterations" << std::endl;
    std::cout << "Optimized implementation: " << optimizedTime << " seconds for "
        << iterations << " iterations" << std::endl;
    std::cout << "\nThroughput:" << std::endl;
    std::cout << "Basic:   " << basicThroughput << " MB/s" << std::endl;
    std::cout << "Optimized: " << optimizedThroughput << " MB/s" << std::endl;
    std::cout << "\nSpeedup: " << speedup << "x" << std::endl;

    return 0;
}
