#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <intrin.h> // 添加CPU特性检测支持
#include <immintrin.h> // 添加AES-NI支持

// SM4 S-Box
static const uint8_t SBOX[256] = {
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xA8,
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

// System parameters (FK)
static const uint32_t FK[4] = {
    0xA3B1BAC6, 0x56AA3350, 0x677D9197, 0xB27022DC
};

// Fixed parameters (CK)
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

// Circular left shift
inline uint32_t ROTL(uint32_t x, uint8_t n) {
    return (x << n) | (x >> (32 - n));
}

// Non-linear transformation tau (S-box substitution)
inline uint32_t tau(uint32_t A) {
    uint32_t B = 0;
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A >> 24)]) << 24;
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A >> 16)]) << 16;
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A >> 8)]) << 8;
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A)]);
    return B;
}

// Linear transformation L for round function
inline uint32_t L(uint32_t B) {
    return B ^ ROTL(B, 2) ^ ROTL(B, 10) ^ ROTL(B, 18) ^ ROTL(B, 24);
}

// Linear transformation L' for key expansion
inline uint32_t L_prime(uint32_t B) {
    return B ^ ROTL(B, 13) ^ ROTL(B, 23);
}

// Combined T function for encryption
inline uint32_t T(uint32_t X) {
    return L(tau(X));
}

// Combined T' function for key schedule
inline uint32_t T_prime(uint32_t X) {
    return L_prime(tau(X));
}

// Key expansion
void SM4_KeySchedule(const uint8_t key[16], uint32_t rk[32], bool forEncryption = true) {
    uint32_t K[36];

    // Initialize intermediate key
    for (int i = 0; i < 4; ++i) {
        K[i] = static_cast<uint32_t>(key[4 * i] << 24) |
            static_cast<uint32_t>(key[4 * i + 1] << 16) |
            static_cast<uint32_t>(key[4 * i + 2] << 8) |
            static_cast<uint32_t>(key[4 * i + 3]);
        K[i] ^= FK[i];
    }

    // Generate round keys
    for (int i = 0; i < 32; ++i) {
        const uint32_t X = K[i + 1] ^ K[i + 2] ^ K[i + 3] ^ CK[i];
        K[i + 4] = K[i] ^ T_prime(X);

        // Store round key in correct order (encryption or decryption)
        if (forEncryption) {
            rk[i] = K[i + 4];
        }
        else {
            rk[31 - i] = K[i + 4];
        }
    }
}

// Process one 128-bit block (common for encryption/decryption)
void SM4_CryptBlock(const uint8_t in[16], uint8_t out[16], const uint32_t rk[32]) {
    uint32_t X[36];

    // Load input words (big-endian)
    for (int i = 0; i < 4; ++i) {
        X[i] = static_cast<uint32_t>(in[4 * i] << 24) |
            static_cast<uint32_t>(in[4 * i + 1] << 16) |
            static_cast<uint32_t>(in[4 * i + 2] << 8) |
            static_cast<uint32_t>(in[4 * i + 3]);
    }

    // 32 rounds of SM4
    for (int i = 0; i < 32; ++i) {
        const uint32_t X_next = X[i] ^ T(X[i + 1] ^ X[i + 2] ^ X[i + 3] ^ rk[i]);
        X[i + 4] = X_next;
    }

    // Store output in reverse order (big-endian)
    for (int i = 0; i < 4; ++i) {
        uint32_t word = X[35 - i];
        out[4 * i] = static_cast<uint8_t>(word >> 24);
        out[4 * i + 1] = static_cast<uint8_t>(word >> 16);
        out[4 * i + 2] = static_cast<uint8_t>(word >> 8);
        out[4 * i + 3] = static_cast<uint8_t>(word);
    }
}

// 检测CPU是否支持AES指令集
bool cpu_supports_aes() {
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    return (cpu_info[2] & (1 << 25)) != 0;
}

// ================== 修复的AES-NI优化实现 ==================
inline uint32_t aesni_T(uint32_t X) {
    // 将32位输入拆分为4个字节（大端序）
    alignas(16) uint8_t in_bytes[16] = {
        static_cast<uint8_t>(X >> 24),  // 最高位字节
        static_cast<uint8_t>(X >> 16),
        static_cast<uint8_t>(X >> 8),
        static_cast<uint8_t>(X),        // 最低位字节
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    __m128i data = _mm_load_si128((const __m128i*)in_bytes);

    // 应用AES-NI的S盒替换
    const __m128i zero = _mm_setzero_si128();
    data = _mm_aesenclast_si128(data, zero);

    // 提取结果
    alignas(16) uint8_t out_bytes[16];
    _mm_store_si128((__m128i*)out_bytes, data);

    // 重组为32位字（大端序）
    uint32_t sbox_result = (static_cast<uint32_t>(out_bytes[0]) << 24) |
        (static_cast<uint32_t>(out_bytes[1]) << 16) |
        (static_cast<uint32_t>(out_bytes[2]) << 8) |
        static_cast<uint32_t>(out_bytes[3]);

    // 应用SM4的线性变换L
    return sbox_result ^ ROTL(sbox_result, 2) ^
        ROTL(sbox_result, 10) ^ ROTL(sbox_result, 18) ^
        ROTL(sbox_result, 24);
}

// AES-NI优化的块处理
void SM4_CryptBlock_AESNI(const uint8_t in[16], uint8_t out[16], const uint32_t rk[32]) {
    uint32_t X[36];

    // 加载输入
    for (int i = 0; i < 4; ++i) {
        X[i] = static_cast<uint32_t>(in[4 * i] << 24) |
            static_cast<uint32_t>(in[4 * i + 1] << 16) |
            static_cast<uint32_t>(in[4 * i + 2] << 8) |
            static_cast<uint32_t>(in[4 * i + 3]);
    }

    // 32轮SM4，使用AES-NI加速的T函数
    for (int i = 0; i < 32; ++i) {
        const uint32_t temp = X[i + 1] ^ X[i + 2] ^ X[i + 3] ^ rk[i];
        const uint32_t X_next = X[i] ^ aesni_T(temp);
        X[i + 4] = X_next;
    }

    // 存储输出
    for (int i = 0; i < 4; ++i) {
        uint32_t word = X[35 - i];
        out[4 * i] = static_cast<uint8_t>(word >> 24);
        out[4 * i + 1] = static_cast<uint8_t>(word >> 16);
        out[4 * i + 2] = static_cast<uint8_t>(word >> 8);
        out[4 * i + 3] = static_cast<uint8_t>(word);
    }
}

// 性能测试函数
void test_performance(const char* name,
    void (*crypt_func)(const uint8_t*, uint8_t*, const uint32_t*),
    const uint8_t* plaintext,
    const uint32_t* rk,
    int iterations = 1000000) {
    uint8_t output[16];

    // 预热缓存
    for (int i = 0; i < 1000; i++) {
        crypt_func(plaintext, output, rk);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        crypt_func(plaintext, output, rk);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double time_per_block = static_cast<double>(duration) / iterations;
    double speed_mbps = (16.0 * iterations) / (static_cast<double>(duration) / 1000000.0) / (1024 * 1024);

    std::cout << "[" << name << "]" << std::endl;
    std::cout << "迭代次数: " << iterations << std::endl;
    std::cout << "总耗时: " << duration << " μs" << std::endl;
    std::cout << "每块耗时: " << time_per_block << " μs" << std::endl;
    std::cout << "吞吐量: " << std::fixed << std::setprecision(2)
        << speed_mbps << " MB/s" << std::endl;
    std::cout << "--------------------------------" << std::endl;
}

// Helper function to print hex output
void PrintHex(const char* label, const uint8_t* data, size_t len) {
    std::cout << label;
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(data[i]) << " ";
    }
    std::cout << std::dec << std::endl;
}

int main() {
    // Example key and plaintext
    uint8_t key[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint8_t plaintext[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint8_t ciphertext_basic[16], ciphertext_aesni[16];
    uint8_t decrypted[16];
    uint32_t encrypt_rk[32];
    uint32_t decrypt_rk[32];

    // Generate keys
    SM4_KeySchedule(key, encrypt_rk, true);
    SM4_KeySchedule(key, decrypt_rk, false);

    // 功能验证
    SM4_CryptBlock(plaintext, ciphertext_basic, encrypt_rk);
    SM4_CryptBlock(ciphertext_basic, decrypted, decrypt_rk);

    // Display results
    PrintHex("Key:         ", key, 16);
    PrintHex("Plaintext:   ", plaintext, 16);
    PrintHex("基础实现密文: ", ciphertext_basic, 16);
    PrintHex("解密结果:    ", decrypted, 16);

    if (memcmp(plaintext, decrypted, 16) == 0) {
        std::cout << "Success: Decryption matches original plaintext!" << std::endl;
    }
    else {
        std::cout << "Error: Decryption failed!" << std::endl;
    }

    // 性能测试
    std::cout << "\n性能测试:" << std::endl;

    // 基础实现性能
    test_performance("基础实现", SM4_CryptBlock, plaintext, encrypt_rk);

    // 检查CPU是否支持AES-NI
    if (cpu_supports_aes()) {
        std::cout << "检测到CPU支持AES-NI指令集，启用优化..." << std::endl;

        // 验证AES-NI实现正确性
        SM4_CryptBlock_AESNI(plaintext, ciphertext_aesni, encrypt_rk);
        PrintHex("AES-NI优化密文: ", ciphertext_aesni, 16);

        if (memcmp(ciphertext_basic, ciphertext_aesni, 16) != 0) {
            std::cerr << "\n错误: AES-NI优化实现与基础实现结果不一致!" << std::endl;

            // 详细调试：比较每轮中间结果
            std::cerr << "\n详细调试（前4轮）:" << std::endl;
            uint32_t X_basic[36];
            uint32_t X_aesni[36];

            // 加载输入
            for (int i = 0; i < 4; ++i) {
                X_basic[i] = X_aesni[i] =
                    static_cast<uint32_t>(plaintext[4 * i] << 24) |
                    static_cast<uint32_t>(plaintext[4 * i + 1] << 16) |
                    static_cast<uint32_t>(plaintext[4 * i + 2] << 8) |
                    static_cast<uint32_t>(plaintext[4 * i + 3]);
            }

            for (int i = 0; i < 4; ++i) {
                const uint32_t temp_basic = X_basic[i + 1] ^ X_basic[i + 2] ^ X_basic[i + 3] ^ encrypt_rk[i];
                const uint32_t temp_aesni = X_aesni[i + 1] ^ X_aesni[i + 2] ^ X_aesni[i + 3] ^ encrypt_rk[i];

                const uint32_t T_basic = T(temp_basic);
                const uint32_t T_aesni = aesni_T(temp_aesni);

                X_basic[i + 4] = X_basic[i] ^ T_basic;
                X_aesni[i + 4] = X_aesni[i] ^ T_aesni;

                std::cerr << "轮次 " << i << ":" << std::endl;
                std::cerr << "  输入: " << std::hex << temp_basic << std::endl;
                std::cerr << "  基础T: " << T_basic << std::endl;
                std::cerr << "  AES-NI T: " << T_aesni << std::endl;
                std::cerr << "  基础输出: " << X_basic[i + 4] << std::endl;
                std::cerr << "  AES-NI输出: " << X_aesni[i + 4] << std::endl;

                if (T_basic != T_aesni) {
                    std::cerr << "  *** T函数不匹配! ***" << std::endl;
                }
                if (X_basic[i + 4] != X_aesni[i + 4]) {
                    std::cerr << "  *** 输出不匹配! ***" << std::endl;
                }
            }
            return 1;
        }
        else {
            std::cout << "验证通过: AES-NI优化实现与基础实现结果一致" << std::endl;
        }

        // AES-NI优化实现性能
        test_performance("AES-NI优化", SM4_CryptBlock_AESNI, plaintext, encrypt_rk);

        // 计算加速比
        const int perf_iterations = 1000000;
        uint8_t temp1[16], temp2[16];

        auto start_basic = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iterations; i++) {
            SM4_CryptBlock(plaintext, temp1, encrypt_rk);
        }
        auto end_basic = std::chrono::high_resolution_clock::now();

        auto start_aesni = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < perf_iterations; i++) {
            SM4_CryptBlock_AESNI(plaintext, temp2, encrypt_rk);
        }
        auto end_aesni = std::chrono::high_resolution_clock::now();

        double time_basic = std::chrono::duration_cast<std::chrono::microseconds>(end_basic - start_basic).count();
        double time_aesni = std::chrono::duration_cast<std::chrono::microseconds>(end_aesni - start_aesni).count();
        double speedup = time_basic / time_aesni;

        std::cout << "性能对比结果:" << std::endl;
        std::cout << "基础实现总耗时: " << time_basic << " μs" << std::endl;
        std::cout << "AES-NI优化总耗时: " << time_aesni << " μs" << std::endl;
        std::cout << "加速比: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    else {
        std::cout << "警告: CPU不支持AES-NI指令集，无法启用优化!" << std::endl;
    }

    return 0;
}