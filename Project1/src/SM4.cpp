#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>

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
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A >>  8)]) <<  8;
    B |= static_cast<uint32_t>(SBOX[static_cast<uint8_t>(A      )]);
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
        K[i] = static_cast<uint32_t>(key[4*i    ] << 24) |
               static_cast<uint32_t>(key[4*i + 1] << 16) |
               static_cast<uint32_t>(key[4*i + 2] <<  8) |
               static_cast<uint32_t>(key[4*i + 3]);
        K[i] ^= FK[i];
    }
    
    // Generate round keys
    for (int i = 0; i < 32; ++i) {
        const uint32_t X = K[i+1] ^ K[i+2] ^ K[i+3] ^ CK[i];
        K[i+4] = K[i] ^ T_prime(X);
        
        // Store round key in correct order (encryption or decryption)
        if (forEncryption) {
            rk[i] = K[i+4];
        } else {
            rk[31 - i] = K[i+4];
        }
    }
}

// Process one 128-bit block (common for encryption/decryption)
void SM4_CryptBlock(const uint8_t in[16], uint8_t out[16], const uint32_t rk[32]) {
    uint32_t X[36];
    
    // Load input words (big-endian)
    for (int i = 0; i < 4; ++i) {
        X[i] = static_cast<uint32_t>(in[4*i    ] << 24) |
               static_cast<uint32_t>(in[4*i + 1] << 16) |
               static_cast<uint32_t>(in[4*i + 2] <<  8) |
               static_cast<uint32_t>(in[4*i + 3]);
    }
    
    // 32 rounds of SM4
    for (int i = 0; i < 32; ++i) {
        const uint32_t X_next = X[i] ^ T(X[i+1] ^ X[i+2] ^ X[i+3] ^ rk[i]);
        X[i+4] = X_next;
    }
    
    // Store output in reverse order (big-endian)
    for (int i = 0; i < 4; ++i) {
        uint32_t word = X[35 - i];
        out[4*i    ] = static_cast<uint8_t>(word >> 24);
        out[4*i + 1] = static_cast<uint8_t>(word >> 16);
        out[4*i + 2] = static_cast<uint8_t>(word >>  8);
        out[4*i + 3] = static_cast<uint8_t>(word);
    }
}

// Encryption function
void SM4_Encrypt(const uint8_t in[16], uint8_t out[16], const uint32_t rk[32]) {
    SM4_CryptBlock(in, out, rk);
}

// Decryption function (uses key in reverse order)
void SM4_Decrypt(const uint8_t in[16], uint8_t out[16], const uint32_t rk[32]) {
    SM4_CryptBlock(in, out, rk);
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
    // Example key and plaintext from GB/T 32907-2016
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
    uint32_t encrypt_rk[32];
    uint32_t decrypt_rk[32];
    
    // Generate keys for encryption and decryption
    SM4_KeySchedule(key, encrypt_rk, true);      // Encryption keys
    SM4_KeySchedule(key, decrypt_rk, false);     // Decryption keys
    
    // Encrypt then decrypt
    SM4_Encrypt(plaintext, ciphertext, encrypt_rk);
    SM4_Decrypt(ciphertext, decrypted, decrypt_rk);
    
    // Display results
    PrintHex("Key:         ", key, 16);
    PrintHex("Plaintext:   ", plaintext, 16);
    PrintHex("Ciphertext:  ", ciphertext, 16);
    PrintHex("Decrypted:   ", decrypted, 16);
    
    // Verify decryption
    if (memcmp(plaintext, decrypted, 16) == 0) {
        std::cout << "Success: Decryption matches original plaintext!" << std::endl;
    } else {
        std::cout << "Error: Decryption failed!" << std::endl;
    }
    
    return 0;
}