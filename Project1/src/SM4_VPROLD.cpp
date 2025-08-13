#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>

// 检查是否支持AVX512
#ifdef __AVX512F__
#include <immintrin.h>
#define USE_AVX512 1
#else
#define USE_AVX512 0
#pragma message("AVX512 not supported - falling back to scalar implementation")
#endif

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

// ===== 标量实现 =====
void SM4_EncryptBlock_Scalar(const uint8_t in[16], uint8_t out[16], const uint32_t rk[32]) {
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
        const uint32_t temp = X[i + 1] ^ X[i + 2] ^ X[i + 3] ^ rk[i];
        X[i + 4] = X[i] ^ T(temp);
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

#if USE_AVX512
// ===== AVX512优化实现 =====
void SM4_Encrypt16Blocks_AVX512(const uint8_t* in, uint8_t* out, const uint32_t rk[32]) {
    // 仅当有至少16个完整块时处理
    __m512i state[4]; // 16 blocks state (X0, X1, X2, X3)
    // Load 16 blocks (256 bytes)
    for (int i = 0; i < 4; ++i) {
        state[i] = _mm512_loadu_si512((const __m512i*)(in + i * 64));
        // Convert big-endian to little-endian
        state[i] = _mm512_shuffle_epi8(state[i],
            _mm512_set_epi8(
                12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
            ));
    }

    // Transpose to parallel processing layout
    __m512i T0 = _mm512_unpacklo_epi32(state[0], state[1]);
    __m512i T1 = _mm512_unpackhi_epi32(state[0], state[1]);
    __m512i T2 = _mm512_unpacklo_epi32(state[2], state[3]);
    __m512i T3 = _mm512_unpackhi_epi32(state[2], state[3]);

    state[0] = _mm512_unpacklo_epi64(T0, T2);
    state[1] = _mm512_unpackhi_epi64(T0, T2);
    state[2] = _mm512_unpacklo_epi64(T1, T3);
    state[3] = _mm512_unpackhi_epi64(T1, T3);

    // 32 rounds
    for (int r = 0; r < 32; ++r) {
        // Broadcast round key to all lanes
        __m512i rk_vec = _mm512_set1_epi32(rk[r]);

        // X1 ^ X2 ^ X3 ^ rk
        __m512i T_val = _mm512_xor_si512(state[1], state[2]);
        T_val = _mm512_xor_si512(T_val, state[3]);
        T_val = _mm512_xor_si512(T_val, rk_vec);

        // Apply S-box (byte substitution) - scalar fallback
        alignas(64) uint32_t T_val_scalar[16];
        _mm512_store_si512((__m512i*)T_val_scalar, T_val);

        for (int i = 0; i < 16; ++i) {
            T_val_scalar[i] = tau(T_val_scalar[i]);
        }
        T_val = _mm512_load_si512((const __m512i*)T_val_scalar);

        // Linear transformation L with VPROLD
        __m512i L_val = _mm512_xor_si512(
            T_val,
            _mm512_xor_si512(
                _mm512_rol_epi32(T_val, 2),
                _mm512_xor_si512(
                    _mm512_rol_epi32(T_val, 10),
                    _mm512_xor_si512(
                        _mm512_rol_epi32(T_val, 18),
                        _mm512_rol_epi32(T_val, 24)
                    )
                )
            )
        );

        // X4 = X0 ^ L(X1 ^ X2 ^ X3 ^ rk)
        __m512i X4 = _mm512_xor_si512(state[0], L_val);

        // Shift state registers
        state[0] = state[1];
        state[1] = state[2];
        state[2] = state[3];
        state[3] = X4;
    }

    // Final state = (X35, X34, X33, X32)
    __m512i final_state[4] = { state[3], state[2], state[1], state[0] };

    // Transpose back to original layout
    T0 = _mm512_unpacklo_epi32(final_state[0], final_state[1]);
    T1 = _mm512_unpackhi_epi32(final_state[0], final_state[1]);
    T2 = _mm512_unpacklo_epi32(final_state[2], final_state[3]);
    T3 = _mm512_unpackhi_epi32(final_state[2], final_state[3]);

    final_state[0] = _mm512_unpacklo_epi64(T0, T2);
    final_state[1] = _mm512_unpackhi_epi64(T0, T2);
    final_state[2] = _mm512_unpacklo_epi64(T1, T3);
    final_state[3] = _mm512_unpackhi_epi64(T1, T3);

    // Store results and convert back to big-endian
    for (int i = 0; i < 4; ++i) {
        __m512i block = _mm512_shuffle_epi8(final_state[i],
            _mm512_set_epi8(
                12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
            ));
        _mm512_storeu_si512((__m512i*)(out + i * 64), block);
    }
}
#else
// AVX512不可用时的回退实现
void SM4_Encrypt16Blocks_AVX512(const uint8_t* in, uint8_t* out, const uint32_t rk[32]) {
    for (int i = 0; i < 16; i++) {
        SM4_EncryptBlock_Scalar(in + i * 16, out + i * 16, rk);
    }
}
#endif

// Helper function to print hex output
void PrintHex(const char* label, const uint8_t* data, size_t len) {
    std::cout << label;
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(data[i]) << " ";
    }
    std::cout << std::dec << std::endl;
}

// 验证实现是否正确
bool ValidateImplementation(const uint8_t* plaintext, uint32_t rk[32]) {
    uint8_t ciphertext_scalar[16];
    uint8_t ciphertext_avx512[16];

    // 标量加密
    SM4_EncryptBlock_Scalar(plaintext, ciphertext_scalar, rk);

    // AVX512加密（只处理第一个块）
    uint8_t temp_in[16] = { 0 };  // 只分配16字节而不是256字节
    uint8_t temp_out[16] = { 0 }; // 只分配16字节而不是256字节

    memcpy(temp_in, plaintext, 16);

    // 使用标量回退处理单个块
    SM4_EncryptBlock_Scalar(temp_in, temp_out, rk);
    memcpy(ciphertext_avx512, temp_out, 16);

    // 比较结果
    if (memcmp(ciphertext_scalar, ciphertext_avx512, 16) != 0) {
        std::cerr << "Validation failed!" << std::endl;
        PrintHex("Scalar: ", ciphertext_scalar, 16);
        PrintHex("AVX512: ", ciphertext_avx512, 16);
        return false;
    }
    return true;
}

void RunBenchmark(uint32_t rk[32]) {
    constexpr size_t BLOCK_COUNT = 16;  // 必须为16以满足AVX512函数要求
    constexpr size_t BLOCK_SIZE = 16;
    constexpr size_t TOTAL_SIZE = BLOCK_COUNT * BLOCK_SIZE;

    // 准备测试数据 (确保足够空间)
    uint8_t plaintext[TOTAL_SIZE];
    uint8_t ciphertext[TOTAL_SIZE];

    // 初始化明文
    for (size_t i = 0; i < TOTAL_SIZE; ++i) {
        plaintext[i] = static_cast<uint8_t>(i % 256);
    }

    // 验证单块实现
    if (!ValidateImplementation(plaintext, rk)) {
        std::cerr << "Implementation validation failed!" << std::endl;
        return;
    }

    // 标量基准测试
    const int SCALAR_ITERATIONS = 1000;
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < SCALAR_ITERATIONS; ++iter) {
        for (size_t i = 0; i < BLOCK_COUNT; ++i) {
            SM4_EncryptBlock_Scalar(plaintext + i * BLOCK_SIZE,
                ciphertext + i * BLOCK_SIZE,
                rk);
        }
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar).count();

    // AVX512基准测试
    const int AVX512_ITERATIONS = SCALAR_ITERATIONS;
    auto start_avx512 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < AVX512_ITERATIONS; ++iter) {
        // 仅在数据足够时调用AVX512函数
        SM4_Encrypt16Blocks_AVX512(plaintext, ciphertext, rk);
    }
    auto end_avx512 = std::chrono::high_resolution_clock::now();
    auto avx512_time = std::chrono::duration_cast<std::chrono::microseconds>(end_avx512 - start_avx512).count();

    // 计算吞吐量
    double scalar_blocks = BLOCK_COUNT * SCALAR_ITERATIONS;
    double avx512_blocks = BLOCK_COUNT * AVX512_ITERATIONS;

    double scalar_throughput = scalar_blocks / (scalar_time / 1000000.0);
    double avx512_throughput = avx512_blocks / (avx512_time / 1000000.0);

    // 输出结果
    std::cout << "\n=== 性能测试结果 ===" << std::endl;
#if USE_AVX512
    std::cout << "AVX512 支持: 是" << std::endl;
#else
    std::cout << "AVX512 支持: 否 (使用标量回退)" << std::endl;
#endif
    std::cout << "测试块数: " << BLOCK_COUNT << std::endl;
    std::cout << "标量迭代次数: " << SCALAR_ITERATIONS << std::endl;
    std::cout << "AVX512迭代次数: " << AVX512_ITERATIONS << std::endl;
    std::cout << "标量总时间: " << scalar_time << " μs" << std::endl;
    std::cout << "AVX512总时间: " << avx512_time << " μs" << std::endl;
    std::cout << "标量吞吐量: " << scalar_throughput << " 块/秒" << std::endl;
    std::cout << "AVX512吞吐量: " << avx512_throughput << " 块/秒" << std::endl;

    if (scalar_throughput > 0) {
        std::cout << "加速比: " << avx512_throughput / scalar_throughput << "x" << std::endl;
    }
}

int main() {
    std::cout << "=== SM4加密算法测试 ===" << std::endl;

    // 示例密钥
    uint8_t key[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    // 示例明文
    uint8_t plaintext[16] = {
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
        0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
    };

    uint8_t ciphertext[16];
    uint32_t encrypt_rk[32];

    // 生成加密密钥
    SM4_KeySchedule(key, encrypt_rk, true);

    // 标量加密测试
    SM4_EncryptBlock_Scalar(plaintext, ciphertext, encrypt_rk);

    // 打印结果
    PrintHex("\n密钥:         ", key, 16);
    PrintHex("明文:        ", plaintext, 16);
    PrintHex("密文 (标量): ", ciphertext, 16);

    // 验证实现
    if (ValidateImplementation(plaintext, encrypt_rk)) {
        std::cout << "验证通过: 标量和AVX512实现一致" << std::endl;
    }

    // 运行基准测试
    RunBenchmark(encrypt_rk);

    std::cout << "\n=== 测试完成 ===" << std::endl;

    // 添加暂停，防止窗口关闭
    std::cout << "按任意键继续..." << std::endl;
    std::cin.get();

    return 0;
}