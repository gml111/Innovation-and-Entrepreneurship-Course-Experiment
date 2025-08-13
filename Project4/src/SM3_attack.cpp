/*#include <iostream>
#include <string>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>
using namespace std;

// 确保字符串长度为指定值，不足则补0
string padString(const string& s, int length) {
    if (s.size() >= length) return s.substr(0, length);
    return string(length - s.size(), '0') + s;
}

// 二进制转换为十六进制函数
string BinToHex(string str) {
    string hex = "";
    int temp = 0;
    while (str.size() % 4 != 0) {
        str = "0" + str;
    }
    for (size_t i = 0; i < str.size(); i += 4) {
        if (i + 3 >= str.size()) break; // 防止越界
        temp = (str[i] - '0') * 8 + (str[i + 1] - '0') * 4 +
            (str[i + 2] - '0') * 2 + (str[i + 3] - '0') * 1;
        if (temp < 10) {
            hex += to_string(temp);
        }
        else {
            hex += 'a' + (temp - 10);
        }
    }
    return hex;
}

// 十六进制转换为二进制函数
string HexToBin(string str) {
    string bin = "";
    const string table[16] = { "0000","0001","0010","0011","0100","0101","0110","0111",
                             "1000","1001","1010","1011","1100","1101","1110","1111" };
    for (char c : str) {
        int idx;
        if (c >= '0' && c <= '9') {
            idx = c - '0';
        }
        else if (c >= 'a' && c <= 'f') {
            idx = 10 + c - 'a';
        }
        else if (c >= 'A' && c <= 'F') {
            idx = 10 + c - 'A';
        }
        else {
            idx = 0; // 无效字符处理
        }
        bin += table[idx];
    }
    return bin;
}

// 十进制转换为十六进制函数（补全指定宽度）
string DecToHex(uint64_t num, int width = 8) {
    stringstream ss;
    ss << hex << setw(width) << setfill('0') << num;
    string res = ss.str();
    transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

// 字符串转换为十六进制字符串
string strToHex(const string& input) {
    stringstream ss;
    ss << hex << setfill('0');
    for (unsigned char c : input) {
        ss << setw(2) << static_cast<int>(c);
    }
    return ss.str();
}

// 循环左移
string LeftShift(string str, int len) {
    str = padString(str, 8); // 确保8位十六进制
    string bin = HexToBin(str);
    bin = padString(bin, 32); // 确保32位二进制
    len = len % 32; // 防止移位过大
    if (len < 0) len += 32; // 处理负移位
    string shifted = bin.substr(len) + bin.substr(0, len);
    return BinToHex(shifted);
}

// 异或操作
string XOR(string str1, string str2) {
    str1 = padString(str1, 8);
    str2 = padString(str2, 8);
    string bin1 = HexToBin(str1);
    string bin2 = HexToBin(str2);
    bin1 = padString(bin1, 32);
    bin2 = padString(bin2, 32);

    string res;
    for (size_t i = 0; i < 32; ++i) {
        res += (bin1[i] != bin2[i]) ? '1' : '0';
    }
    return BinToHex(res);
}

// 与操作
string AND(string str1, string str2) {
    str1 = padString(str1, 8);
    str2 = padString(str2, 8);
    string bin1 = HexToBin(str1);
    string bin2 = HexToBin(str2);
    bin1 = padString(bin1, 32);
    bin2 = padString(bin2, 32);

    string res;
    for (size_t i = 0; i < 32; ++i) {
        res += (bin1[i] == '1' && bin2[i] == '1') ? '1' : '0';
    }
    return BinToHex(res);
}

// 或操作
string OR(string str1, string str2) {
    str1 = padString(str1, 8);
    str2 = padString(str2, 8);
    string bin1 = HexToBin(str1);
    string bin2 = HexToBin(str2);
    bin1 = padString(bin1, 32);
    bin2 = padString(bin2, 32);

    string res;
    for (size_t i = 0; i < 32; ++i) {
        res += (bin1[i] == '1' || bin2[i] == '1') ? '1' : '0';
    }
    return BinToHex(res);
}

// 非操作
string NOT(string str) {
    str = padString(str, 8);
    string bin = HexToBin(str);
    bin = padString(bin, 32);

    string res;
    for (char c : bin) {
        res += (c == '0') ? '1' : '0';
    }
    return BinToHex(res);
}

// 模加运算（模2^32）
string ModAdd(string a, string b) {
    a = padString(a, 8);
    b = padString(b, 8);

    unsigned long long a_val = stoull(a, nullptr, 16);
    unsigned long long b_val = stoull(b, nullptr, 16);
    unsigned long long result = (a_val + b_val) % 0x100000000ULL;

    return DecToHex(static_cast<uint64_t>(result));
}

// 置换函数P0
string P0(string str) {
    return XOR(XOR(str, LeftShift(str, 9)), LeftShift(str, 17));
}

// 置换函数P1
string P1(string str) {
    return XOR(XOR(str, LeftShift(str, 15)), LeftShift(str, 23));
}

// Tj常量
string T(int j) {
    if (j >= 0 && j <= 15) {
        return "79cc4519";
    }
    else {
        return "7a879d8a";
    }
}

// 布尔函数FF
string FF(string X, string Y, string Z, int j) {
    if (j >= 0 && j <= 15) {
        return XOR(XOR(X, Y), Z);
    }
    else {
        return OR(OR(AND(X, Y), AND(X, Z)), AND(Y, Z));
    }
}

// 布尔函数GG
string GG(string X, string Y, string Z, int j) {
    if (j >= 0 && j <= 15) {
        return XOR(XOR(X, Y), Z);
    }
    else {
        return OR(AND(X, Y), AND(NOT(X), Z));
    }
}

// 消息扩展函数
string extension(string str) {
    // 确保输入是128字符（512位）
    if (str.size() < 128) {
        str += string(128 - str.size(), '0');
    }
    else if (str.size() > 128) {
        str = str.substr(0, 128);
    }

    string W = str;
    // 生成W16到W67
    for (int j = 16; j < 68; ++j) {
        if ((j - 16) * 8 + 8 > W.size()) break;
        string w16 = W.substr((j - 16) * 8, 8);
        string w9 = W.substr((j - 9) * 8, 8);
        string w3 = W.substr((j - 3) * 8, 8);
        string w13 = W.substr((j - 13) * 8, 8);
        string w6 = W.substr((j - 6) * 8, 8);

        string part1 = XOR(w16, w9);
        string part2 = LeftShift(w3, 15);
        string part3 = XOR(part1, part2);
        string part4 = P1(part3);
        string part5 = LeftShift(w13, 7);
        string wj = XOR(XOR(part4, part5), w6);
        W += wj;
    }

    // 生成W'0到W'63
    for (int j = 0; j < 64; ++j) {
        if (j * 8 + 8 > W.size() || (j + 4) * 8 + 8 > W.size()) break;
        string wj = XOR(W.substr(j * 8, 8), W.substr((j + 4) * 8, 8));
        W += wj;
    }

    return W;
}

// 压缩函数
string compress(string W, string IV) {
    // 确保IV是64字符（256位）
    IV = padString(IV, 64);

    string A = IV.substr(0, 8);
    string B = IV.substr(8, 8);
    string C = IV.substr(16, 8);
    string D = IV.substr(24, 8);
    string E = IV.substr(32, 8);
    string F = IV.substr(40, 8);
    string G = IV.substr(48, 8);
    string H = IV.substr(56, 8);

    for (int j = 0; j < 64; ++j) {
        // 确保W有足够的长度
        if ((j + 68) * 8 + 8 > W.size() || j * 8 + 8 > W.size()) break;

        // 计算SS1
        string shlA12 = LeftShift(A, 12);
        string t = T(j);
        string shlT = LeftShift(t, j % 32);
        string add1 = ModAdd(shlA12, E);
        string add2 = ModAdd(add1, shlT);
        string SS1 = LeftShift(add2, 7);

        // 计算SS2
        string SS2 = XOR(SS1, shlA12);

        // 计算TT1
        string ff = FF(A, B, C, j);
        string add3 = ModAdd(ff, D);
        string add4 = ModAdd(add3, SS2);
        string wp = W.substr((j + 68) * 8, 8); // W'j
        string TT1 = ModAdd(add4, wp);

        // 计算TT2
        string gg = GG(E, F, G, j);
        string add5 = ModAdd(gg, H);
        string add6 = ModAdd(add5, SS1);
        string wj = W.substr(j * 8, 8); // Wj
        string TT2 = ModAdd(add6, wj);

        // 更新寄存器
        D = C;
        C = LeftShift(B, 9);
        B = A;
        A = TT1;
        H = G;
        G = LeftShift(F, 19);
        F = E;
        E = P0(TT2);
    }

    return A + B + C + D + E + F + G + H;
}

// 消息填充函数
string padding(const string& msg_hex, uint64_t& total_bit_len) {
    int msg_bit_len = msg_hex.size() * 4;
    total_bit_len = msg_bit_len;

    // 填充规则：1 + k个0 + 64位长度
    int k = (448 - (msg_bit_len + 1) % 512 + 512) % 512;
    string padding_str;

    // 添加'1'位和k个'0'位（转换为十六进制）
    padding_str += "8"; // 二进制1000
    padding_str += string(k / 4, '0'); // 每个十六进制字符4位

    // 添加64位长度（大端序）
    padding_str += DecToHex(total_bit_len, 16);

    return msg_hex + padding_str;
}

// SM3哈希计算
string sm3_hash(const string& msg_hex, string IV = "7380166f4914b2b9172442d7da8a0600a96f30bc163138aae38dee4db0fb0e4e") {
    uint64_t total_len;
    string padded = padding(msg_hex, total_len);
    int num_blocks = padded.size() / 128;
    if (padded.size() % 128 != 0) num_blocks++; // 处理最后一个不完整块

    string V = IV;
    for (int i = 0; i < num_blocks; ++i) {
        int start = i * 128;
        string block = (start + 128 <= padded.size()) ?
            padded.substr(start, 128) :
            padded.substr(start) + string(128 - (padded.size() - start), '0');

        string extended = extension(block);
        V = compress(extended, V);
    }

    return V;
}

// 长度扩展攻击函数
string length_extension_attack(const string& original_hash, int original_bit_len, const string& extension) {
    // 1. 计算原始消息的填充长度
    int k = (448 - (original_bit_len + 1) % 512 + 512) % 512;
    int pad_bit_len = 1 + k + 64;

    // 2. 扩展消息处理
    string ext_hex = strToHex(extension);
    uint64_t ext_total_len = original_bit_len + pad_bit_len + ext_hex.size() * 4;

    // 3. 构造扩展消息的填充
    string ext_padded = ext_hex;
    uint64_t dummy;
    ext_padded = padding(ext_padded, dummy);

    // 4. 使用原始哈希作为初始向量计算
    return sm3_hash(ext_padded, original_hash);
}

int main() {
    try {
        cout << "SM3长度扩展攻击验证" << endl;
        cout << "==========================================" << endl;

        // 原始消息
        string original_msg = "Hello, world!";
        int original_bit_len = original_msg.size() * 8;

        cout << "原始消息: \"" << original_msg << "\"" << endl;
        cout << "原始消息长度: " << original_bit_len << " 比特" << endl;

        // 计算原始哈希
        string original_hex = strToHex(original_msg);
        string original_hash = sm3_hash(original_hex);
        cout << "原始哈希值: " << original_hash << endl << endl;

        // 扩展消息
        string extension_msg = " This is an extension!";
        cout << "扩展消息: \"" << extension_msg << "\"" << endl;

        // 执行长度扩展攻击
        string forged_hash = length_extension_attack(original_hash, original_bit_len, extension_msg);
        cout << "攻击生成的伪造哈希值: " << forged_hash << endl << endl;

        // 计算真实的新消息哈希（原始消息 + 填充 + 扩展消息）
        string original_padded_hex;
        uint64_t dummy_len;
        original_padded_hex = padding(original_hex, dummy_len);
        string new_msg_hex = original_padded_hex + strToHex(extension_msg);
        string real_hash = sm3_hash(new_msg_hex);
        cout << "真实的新消息哈希值: " << real_hash << endl << endl;

        // 验证结果
        if (forged_hash == real_hash) {
            cout << "验证结果: 长度扩展攻击成功!" << endl;
            cout << "攻击者在未知原始消息的情况下，成功伪造了扩展消息的哈希值" << endl;
        }
        else {
            cout << "验证结果: 长度扩展攻击失败!" << endl;
        }
    }
    catch (const exception& e) {
        cerr << "运行时错误: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        cerr << "发生未知错误" << endl;
        return 1;
    }

    return 0;
}



import random
import hashlib
import math

# 模拟的同态加密系统（仅供演示）
class SimpleHomomorphicEncryption :
    def __init__(self) :
    # 生成大质数 p 和 q
    self.p = self.generate_large_prime(1024)
    self.q = self.generate_large_prime(1024)
    self.n = self.p * self.q
    self.nsquare = self.n * self.n

    def generate_large_prime(self, bits) :
    """生成大质数（简化实现）"""
    while True :
        p = random.randint(2 * *(bits - 1), 2 * *bits)
        if p % 2 != 0 and self.is_prime(p) :
            return p

            def is_prime(self, n, k = 10) :
            """米勒-拉宾素数测试"""
            if n <= 1 :
                return False
                if n == 2 :
                    return True
                    if n % 2 == 0 :
                        return False

                        d = n - 1
                        r = 0
                        while d % 2 == 0 :
                            d //= 2
                            r += 1

                            for _ in range(k) :
                                a = random.randint(2, n - 2)
                                x = pow(a, d, n)
                                if x == 1 or x == n - 1 :
                                    continue
                                    for _ in range(r - 1) :
                                        x = pow(x, 2, n)
                                        if x == n - 1 :
                                            break
                                        else :
                                            return False
                                            return True

                                            def encrypt(self, plaintext) :
                                            """模拟加密"""
                                            r = random.randint(1, self.n - 1)
                                            # Paillier 加密公式的简化模拟 : (1 + plaintext * n) * r ^ n mod n ^ 2
                                            ciphertext = (1 + plaintext * self.n) * pow(r, self.n, self.nsquare) % self.nsquare
                                            return { 'ciphertext': ciphertext, 'r' : r }

                                            def decrypt(self, ciphertext) :
                                            """模拟解密"""
                                            # 简化解密过程
                                            return (ciphertext - 1) // self.n

                                            def raw_decrypt(self, ciphertext) :
                                            """直接解密整数密文"""
                                            return (ciphertext - 1) // self.n

                                            # 使用简化的哈希函数
                                            def hash_to_point(u: str)->bytes:
"""将字符串映射为固定长度的哈希值"""
return hashlib.sha256(u.encode()).digest()

# 简化的点运算
def point_operation(value: bytes, key : int)->bytes:
"""使用密钥对哈希值进行处理"""
key_bytes = key.to_bytes(32, 'big')
# 使用异或操作和哈希组合
transformed = bytes([v ^ k for v, k in zip(value, key_bytes)])
return hashlib.sha256(transformed).digest()

class Party1 :
    def __init__(self, set_V: list) :
    self.set_V = set_V
    self.k1 = random.randint(1, 2 * *256 - 1)  # 256位随机密钥
    self.paillier_pubkey = None
    self.intersection_sum_cipher = None

    def set_paillier_pubkey(self, pubkey) :
    self.paillier_pubkey = pubkey

    def round1(self)->list :
    # 计算 H(v_i) ^ k1
    self.hashed_exponents = []
    for v in self.set_V:
P = hash_to_point(v)
Q = point_operation(P, self.k1)
self.hashed_exponents.append(Q)

# 随机洗牌
random.shuffle(self.hashed_exponents)
return self.hashed_exponents

def round3(self, Z: list, received_set : list) :
    # 将Z集合转换为字节串集合
    Z_bytes_set = set(Z)

    # 处理接收的元组[(H(w_j) ^ k2, enc(t_j)]
        intersection_ciphers = []
        for point_bytes, enc_t in received_set :
# 计算 H(w_j)^ { k1 * k2 }
T = point_operation(point_bytes, self.k1)

# 检查是否在Z中
if T in Z_bytes_set :
intersection_ciphers.append(enc_t)

# 同态求和
if intersection_ciphers:
product = 1
nsquare = self.paillier_pubkey.nsquare
# 将所有密文相乘
for c in intersection_ciphers :
product = (product * c) % nsquare

# 刷新密文(重新随机化)
r = random.randint(1, self.paillier_pubkey.n - 1)
c0 = pow(r, self.paillier_pubkey.n, nsquare)  # 加密0
refreshed_cipher = (product * c0) % nsquare
else:
# 如果交集为空，加密0
refreshed_cipher = self.paillier_pubkey.encrypt(0)['ciphertext']

self.intersection_sum_cipher = refreshed_cipher
return refreshed_cipher

class Party2 :
    def __init__(self, set_W: list) :
    self.set_W = set_W
    self.k2 = random.randint(1, 2 * *256 - 1)  # 256位随机密钥
    # 生成同态加密密钥
    self.paillier_public_key = SimpleHomomorphicEncryption()
    self.paillier_private_key = self.paillier_public_key
    self.intersection_sum = None

    def get_paillier_pubkey(self) :
    return self.paillier_public_key

    def round2(self, received_points: list)->tuple :
    # 步骤1 : 计算 H(v_i)^ { k1 * k2 }
Z = []
for P in received_points :
Q = point_operation(P, self.k2)
Z.append(Q)
random.shuffle(Z)

# 步骤2 : 处理自己的集合
processed_set = []
for w, t in self.set_W :
    # 计算 H(w_j)^ { k2 }
P = hash_to_point(w)
Q = point_operation(P, self.k2)
# 加密 t_j
enc_t = self.paillier_public_key.encrypt(t)['ciphertext']
processed_set.append((Q, enc_t))

random.shuffle(processed_set)
return Z, processed_set

def decrypt_sum(self, ciphertext: int) -> int:
# 直接解密整数密文
self.intersection_sum = self.paillier_private_key.raw_decrypt(ciphertext)
return self.intersection_sum

# 测试协议执行
if __name__ == "__main__":
# 示例数据集
set_V = ["id1", "id2", "id3", "id5"]
set_W = [("id1", 10), ("id2", 20), ("id3", 30), ("id4", 40)]

# 初始化参与方
p1 = Party1(set_V)
p2 = Party2(set_W)

# 设置阶段: P2发送公钥给P1
p1.set_paillier_pubkey(p2.get_paillier_pubkey())

# 第1轮 : P1->P2
r1_output = p1.round1()

# 第2轮 : P2->P1(发送Z和处理后的集合)
Z, r2_output = p2.round2(r1_output)

# 第3轮 : P1->P2(发送加密的和)
sum_cipher = p1.round3(Z, r2_output)

# P2解密得到结果
result = p2.decrypt_sum(sum_cipher)

print(f"交集元素: id1, id2, id3 (预期和为60)")
print(f"实际计算得到的交集和: {result}")
















import os
import random
import binascii
from gmssl import sm3, func

# ===== 调试模式配置 =====
DEBUG_FIXED_KEYS = True  # 固定私钥 / 公钥
DEBUG_FIXED_K = True  # 固定签名随机数 k
FIXED_PRIVATE_KEY = int(
    "128D97B99C874D5443E4D2F2A9FA9130EBF2B7E9E1E5D7A441BDCE3D1F2A29AC", 16
    )
    FIXED_K = int(
        "1F1E1D1C1B1A191817161514131211100F0E0D0C0B0A09080706050403020100", 16
        )
# ======================

    # 椭圆曲线参数（SM2推荐参数）
    PRIME = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
    A = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
    B = 0x28E9FA9E9D9F5E344D5AEF7F6BFFFF5F
    Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
    Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
    N = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
    G = (Gx, Gy)


    def mod_inv(a, modulus = PRIME) :
    """扩展欧几里得算法求模逆"""
    if a == 0 :
        return 0
        lm, hm = 1, 0
        low, high = a % modulus, modulus
        while low > 1:
r = high // low
nm, new = hm - lm * r, high - low * r
lm, low, hm, high = nm, new, lm, low
return lm % modulus


def elliptic_add(point1, point2) :
    """椭圆曲线点加法"""
    if point1[0] == point2[0] and point1[1] == point2[1] :
        return elliptic_double(point1)
        if point1 == (0, 0) :
            return point2
            if point2 == (0, 0) :
                return point1
                if point1[0] == point2[0] :
                    return (0, 0)

                    # 计算斜率
                    dx = (point2[0] - point1[0]) % PRIME
                    dy = (point2[1] - point1[1]) % PRIME
                    s = dy * mod_inv(dx, PRIME)
                    x = (s * *2 - point1[0] - point2[0]) % PRIME
                    y = (s * (point1[0] - x) - point1[1]) % PRIME
                    return (x, y)


                    def elliptic_double(point) :
                    """椭圆曲线倍点运算"""
                    if point == (0, 0) :
                        return point

                        # 计算斜率
                        numerator = (3 * point[0] * *2 + A) % PRIME
                        denominator = (2 * point[1]) % PRIME
                        s = numerator * mod_inv(denominator, PRIME)

                        x = (s * *2 - 2 * point[0]) % PRIME
                        y = (s * (point[0] - x) - point[1]) % PRIME
                        return (x, y)


                        def elliptic_multiply(k, point) :
                        """椭圆曲线点乘法（标量乘法）"""
                        if k% N == 0 or point == (0, 0) :
                            return (0, 0)
                            if k < 0 :
                                return elliptic_multiply(-k, point)

                                # 二进制展开
                                result = (0, 0)
                                addend = point

                                while k:
if k & 1 :
    result = elliptic_add(result, addend)
    addend = elliptic_double(addend)
    k >>= 1

    return result


    def get_za(user_id, public_key) :
    """计算Z_A值（SM2预计算哈希部分）"""
    # 用户ID默认值
    if user_id is None :
user_id = "1234567812345678"

entl = len(user_id.encode('utf-8')) * 8
entl_bytes = entl.to_bytes(2, 'big')

# 转换曲线参数到字节
a_bytes = A.to_bytes(32, 'big')
b_bytes = B.to_bytes(32, 'big')
gx_bytes = Gx.to_bytes(32, 'big')
gy_bytes = Gy.to_bytes(32, 'big')

# 转换公钥到字节
px_bytes = public_key[0].to_bytes(32, 'big')
py_bytes = public_key[1].to_bytes(32, 'big')

# 构造Z_A哈希输入
za_input = (
    entl_bytes +
    user_id.encode('utf-8') +
    a_bytes +
    b_bytes +
    gx_bytes +
    gy_bytes +
    px_bytes +
    py_bytes
    )

    # 计算哈希并返回十六进制字符串
    return sm3.sm3_hash(func.bytes_to_list(za_input))


    class SM2 :
    def __init__(self, private_key = None, public_key = None, user_id = None) :
    self.user_id = user_id

    if DEBUG_FIXED_KEYS :
        self.private_key = FIXED_PRIVATE_KEY
        self.public_key = elliptic_multiply(self.private_key, G)
    else:
self.private_key = private_key
if private_key :
    self.public_key = elliptic_multiply(private_key, G)
else :
    self.public_key = public_key if public_key else None

    # 计算并存储ZA
    self.za = get_za(self.user_id, self.public_key) if self.public_key else None

    def generate_key_pair(self) :
    """生成密钥对"""
    self.private_key = random.randint(1, N - 1)
    self.public_key = elliptic_multiply(self.private_key, G)
    self.za = get_za(self.user_id, self.public_key)
    return self.private_key, self.public_key

    def sign(self, message) :
    """SM2签名"""
    if not self.private_key :
        raise ValueError("Private key is required for signing")

        e = self._hash_message(message)
        if DEBUG_FIXED_K :
            k = FIXED_K % N
        else:
k = random.randint(1, N - 1)

# 计算椭圆曲线点[k]G
x1, y1 = elliptic_multiply(k, G)
r = (e + x1) % N

# 避免r为0或r + k == n的情况
if r == 0 or r + k == N:
return self.sign(message)

# 计算s = (1 + d)^ { -1 } *(k - r * d) mod n
d = self.private_key
s = (pow(1 + d, N - 2, N) * (k - r * d)) % N

# 避免s为0
if s == 0:
return self.sign(message)

return (r, s)

def verify(self, message, signature) :
    """SM2验证签名"""
    if not self.public_key :
        raise ValueError("Public key is required for verification")

        r, s = signature

        # 检查r和s是否在[1, n - 1]范围内
        if not (1 <= r <= N - 1 and 1 <= s <= N - 1) :
            return False

            e = self._hash_message(message)
            t = (r + s) % N

            # 计算椭圆曲线点[s]G + [t]P
            point1 = elliptic_multiply(s, G)
            point2 = elliptic_multiply(t, self.public_key)
            x1, y1 = elliptic_add(point1, point2)

            # 验证R = (e + x1) mod n
            return (r % N) == ((e + x1) % N)

            def _hash_message(self, message) :
            """计算消息哈希e = Hash(Z_A || M)"""
            if not self.za :
                raise ValueError("ZA not initialized")

                msg_bytes = message.encode('utf-8')
                # 将ZA（十六进制字符串）转换为字节数组
                za_bytes = bytes.fromhex(self.za)
                input_data = func.bytes_to_list(za_bytes + msg_bytes)
                hash_hex = sm3.sm3_hash(input_data)
                return int(hash_hex, 16)


# ======== 测试用例 ========
                if __name__ == "__main__":
# 1. 密钥生成测试
print("===== SM2算法实现测试 =====")
sm2 = SM2(user_id = "test@example.com")
print("公钥生成成功:", (hex(sm2.public_key[0])[:16] + "...", hex(sm2.public_key[1])[:16] + "..."))

# 2. 签名测试
message = "Hello, SM2!"
signature = sm2.sign(message)
print(f"\n消息 '{message}' 的签名:")
print(f"r: {hex(signature[0])}")
print(f"s: {hex(signature[1])}")

# 3. 验证测试
verify_result = sm2.verify(message, signature)
print(f"\n签名验证结果: {'成功' if verify_result else '失败'}")

# 4. 篡改消息验证
tampered_message = "Hello, SM3!"
verify_result_tampered = sm2.verify(tampered_message, signature)
print(f"篡改消息验证结果: {'成功' if verify_result_tampered else '失败'}")

# 5. 使用不同用户ID测试
print("\n===== 不同用户ID测试 =====")
sm2_custom = SM2(user_id = "Alice@email.com")
signature_custom = sm2_custom.sign(message)
verify_result_custom = sm2_custom.verify(message, signature_custom)
print(f"自定义ID签名验证: {'成功' if verify_result_custom else '失败'}")

# 验证不同用户ID的签名是否相同
same_signature = signature == signature_custom
print(f"不同用户ID的签名是否相同: {'是' if same_signature else '否'}")

print("\n所有测试完成!")

























import random
from math import gcd, ceil, log
from gmssl import sm3, func
import binascii

# ===== 调试模式配置 =====
DEBUG_FIXED_KEYS = True  # 固定私钥 / 公钥
DEBUG_FIXED_K = True  # 固定签名随机数 k
FIXED_PRIVATE_KEY = int(
    "128B2FA8BD433C6C068C8D803DFF79792A519A55171B1B650C23661D15897263", 16
    )
    FIXED_K = int(
        "6CB28D99385C175C94F94E934817663FC176D925DD72B727260DBAAE1FB2F96F", 16
        )
# ======================

    # 椭圆曲线参数（与成功案例保持一致）
    PRIME = 0x8542D69E4C044F18E8B92435BF6FF7DE457283915C45517D722EDB8B08F1DFC3
    A = 0x787968B4FA32C3FD2417842E73BBFEFF2F3C848B6831D7E0EC65228B3937E498
    B = 0x63E4C6D3B23B0C849CF84241484BFE48F61D59A5B16BA06E6E12D1DA27C5249A
    Gx = 0x421DEBD61B62EAB6746434EBC3CC315E32220B3BADD50BDC4C4E6C147FEDD43D
    Gy = 0x0680512BCBB42C07D47349D2153B70C4E5D7FDFCBFA36EA1A85841B9E46E09A2
    N = 0x8542D69E4C044F18E8B92435BF6FF7DD297720630485628D5AE74EE7C32E79B7
    G = (Gx, Gy)


    def mod_inv(a, modulus = N) :
    """扩展欧几里得算法求模逆，使用N作为默认模数"""
    if a == 0 :
        return 0
        lm, hm = 1, 0
        low, high = a % modulus, modulus
        while low > 1:
r = high // low
nm, new = hm - lm * r, high - low * r
lm, low, hm, high = nm, new, lm, low
return lm % modulus


def elliptic_add(point1, point2) :
    """椭圆曲线点加法"""
    if point1 == (0, 0) :
        return point2
        if point2 == (0, 0) :
            return point1
            if point1 == point2 :
                return elliptic_double(point1)
                if point1[0] == point2[0] :
                    return (0, 0)  # 无穷远点

                    # 计算斜率
                    dx = (point2[0] - point1[0]) % PRIME
                    dy = (point2[1] - point1[1]) % PRIME
                    s = dy * mod_inv(dx, PRIME) % PRIME
                    x = (s * *2 - point1[0] - point2[0]) % PRIME
                    y = (s * (point1[0] - x) - point1[1]) % PRIME
                    return (x, y)


                    def elliptic_double(point) :
                    """椭圆曲线倍点运算"""
                    if point == (0, 0) :
                        return point

                        # 计算斜率
                        numerator = (3 * point[0] * *2 + A) % PRIME
                        denominator = (2 * point[1]) % PRIME
                        s = numerator * mod_inv(denominator, PRIME) % PRIME

                        x = (s * *2 - 2 * point[0]) % PRIME
                        y = (s * (point[0] - x) - point[1]) % PRIME
                        return (x, y)


                        def elliptic_multiply(k, point) :
                        """椭圆曲线点乘法（标量乘法）"""
                        if k% N == 0 or point == (0, 0) :
                            return (0, 0)
                            if k < 0 :
                                return elliptic_multiply(-k, point)

                                # 二进制展开法
                                result = (0, 0)
                                addend = point

                                while k:
if k & 1 :
    result = elliptic_add(result, addend)
    addend = elliptic_double(addend)
    k >>= 1

    return result


    def get_za(user_id, public_key) :
    """计算Z_A值（SM2预计算哈希部分）"""
    # 用户ID默认值
    if user_id is None :
user_id = "ALICE123@YAHOO.COM"

entl = len(user_id.encode('utf-8')) * 8
entl_bytes = entl.to_bytes(2, 'big')

# 转换曲线参数到字节
a_bytes = A.to_bytes(32, 'big')
b_bytes = B.to_bytes(32, 'big')
gx_bytes = Gx.to_bytes(32, 'big')
gy_bytes = Gy.to_bytes(32, 'big')

# 转换公钥到字节
px_bytes = public_key[0].to_bytes(32, 'big')
py_bytes = public_key[1].to_bytes(32, 'big')

# 构造Z_A哈希输入
za_input = (
    entl_bytes +
    user_id.encode('utf-8') +
    a_bytes +
    b_bytes +
    gx_bytes +
    gy_bytes +
    px_bytes +
    py_bytes
    )

    # 计算哈希并返回十六进制字符串
    return sm3.sm3_hash(func.bytes_to_list(za_input))


    def compute_hash(za, message) :
    """计算消息哈希e = Hash(Z_A || M)"""
    msg_bytes = message.encode('utf-8')
    za_bytes = bytes.fromhex(za)
    input_data = func.bytes_to_list(za_bytes + msg_bytes)
    hash_hex = sm3.sm3_hash(input_data)
    return int(hash_hex, 16) % N


    class SM2 :
    def __init__(self, private_key = None, public_key = None, user_id = None) :
    self.user_id = user_id or "ALICE123@YAHOO.COM"
    self.curve_params = {
        'p': PRIME,
        'a' : A,
        'b' : B,
        'n' : N,
        'Gx' : Gx,
        'Gy' : Gy
}

if DEBUG_FIXED_KEYS:
self.private_key = FIXED_PRIVATE_KEY
self.public_key = elliptic_multiply(self.private_key, G)
else:
self.private_key = private_key
if private_key :
    self.public_key = elliptic_multiply(private_key, G)
else :
    self.public_key = public_key if public_key else None

    # 计算并存储ZA
    if self.public_key:
self.za = get_za(self.user_id, self.public_key)
    else:
self.za = None

def generate_key_pair(self) :
    """生成密钥对"""
    self.private_key = random.randint(1, N - 1)
    self.public_key = elliptic_multiply(self.private_key, G)
    self.za = get_za(self.user_id, self.public_key)
    return self.private_key, self.public_key

    def sign(self, message, k = None) :
    """SM2签名"""
    if not self.private_key :
        raise ValueError("Private key is required for signing")

        e = compute_hash(self.za, message)

        if k is None :
if DEBUG_FIXED_K :
    k = FIXED_K % N
else :
    k = random.randint(1, N - 1)
        else:
k = k % N

# 计算椭圆曲线点[k]G
x1, y1 = elliptic_multiply(k, G)
r = (e + x1) % N

# 避免r为0或r + k == n的情况
if r == 0 or (r + k) % N == 0:
return self.sign(message, k)

# 计算s = (1 + d)^ { -1 } *(k - r * d) mod n
d = self.private_key
s = mod_inv(1 + d, N) * (k - r * d) % N

# 确保s为正数
if s < 0:
s += N

# 避免s为0
if s == 0:
return self.sign(message, k)

return (r, s)

def verify(self, message, signature, public_key = None, user_id = None) :
    """SM2验证签名"""
    if public_key is None :
public_key = self.public_key
if public_key is None :
raise ValueError("Public key is required for verification")

user_id = user_id or self.user_id
za = get_za(user_id, public_key)

r, s = signature

# 检查r和s是否在[1, n - 1]范围内
if not (1 <= r <= N - 1 and 1 <= s <= N - 1) :
    return False

    e = compute_hash(za, message)
    t = (r + s) % N
    if t == 0 :
        return False

        # 计算椭圆曲线点[s]G + [t]P
        sG = elliptic_multiply(s, G)
        tP = elliptic_multiply(t, public_key)
        x1, y1 = elliptic_add(sG, tP)

        # 验证R = (e + x1) mod n
        R = (e + x1) % N
        return R == r

# ===== 攻击演示方法 =====
        def k_leakage_attack(self, signature, k, message, public_key = None, user_id = None) :
        """已知k值推导私钥的攻击演示"""
        if public_key is None :
public_key = self.public_key
if public_key is None :
raise ValueError("Public key required for attack")

user_id = user_id or self.user_id
za = get_za(user_id, public_key)

r, s = signature
e = compute_hash(za, message)

# 攻击公式：d = (k - s) * inv(s + r) mod n
numerator = (k - s) % N
denominator = (s + r) % N

# 检查分母是否为0
if denominator == 0:
print("错误：计算私钥时遇到零分母，无法计算模逆")
return None

inv_denom = mod_inv(denominator)
if inv_denom is None :
return None

private_key_guess = (numerator * inv_denom) % N
# 确保结果为正数
if private_key_guess < 0:
private_key_guess += N
return private_key_guess

def repeated_k_attack(self, signature1, message1, signature2, message2, public_key = None, user_id = None) :
    """相同k值对多消息签名的攻击演示 - 修复k值推导公式"""
    if public_key is None :
public_key = self.public_key
if public_key is None :
raise ValueError("Public key required for attack")

user_id = user_id or self.user_id
za = get_za(user_id, public_key)

r1, s1 = signature1
r2, s2 = signature2

# 计算两条消息的哈希值
e1 = compute_hash(za, message1)
e2 = compute_hash(za, message2)

# 重新推导k值计算公式：k = (e1 - e2 + s1 * r1 - s2 * r2) * inv(s2 - s1) mod n
numerator_k = (e1 - e2 + s1 * r1 - s2 * r2) % N
denominator_k = (s2 - s1) % N

# 检查分母是否为0
if denominator_k == 0:
print("错误：delta_s为0，无法计算模逆")
return None

inv_denominator_k = mod_inv(denominator_k)
if inv_denominator_k is None :
return None

k_guess = (numerator_k * inv_denominator_k) % N

# 推导私钥
private_key_guess = self.k_leakage_attack(signature1, k_guess, message1, public_key, user_id)
return private_key_guess

def multi_user_k_attack(self, signature, message, public_key, user_id, k) :
    """多个用户使用相同k值的攻击演示 - 修复私钥推导公式"""
    # 计算Z_A
    za = get_za(user_id, public_key)

    r, s = signature
    e = compute_hash(za, message)

    # 正确的私钥推导公式: d = (k - s * (1 + d)) / r mod n
    # 重新整理为 : d = (k - s) * inv(r + s) mod n
    numerator = (k - s) % N
    denominator = (r + s) % N

    # 确保分母为正数
    if denominator < 0:
denominator += N

# 检查分母是否为0
if denominator == 0:
print("错误：计算私钥时遇到零分母，无法计算模逆")
return None

inv_denom = mod_inv(denominator)
if inv_denom is None :
return None

private_key_guess = (numerator * inv_denom) % N
# 确保结果为正数
if private_key_guess < 0:
private_key_guess += N
return private_key_guess


if __name__ == "__main__" :
    print("SM2数字签名算法与攻击演示".center(80, '='))

    # 创建SM2实例
    sm2 = SM2()

    # 生成密钥对
    print("\n生成密钥对...")
    private_key = FIXED_PRIVATE_KEY
    public_key = elliptic_multiply(private_key, G)
    print(f"私钥: {hex(private_key)}")
    print(f"公钥: (x={hex(public_key[0])[:20]}..., y={hex(public_key[1])[:20]}...)")

    # 原始消息
    message = "message digest"
    print(f"\n原始消息: '{message}'")

    # 验证签名功能
    print("\n===== 验证签名功能 =====")
    signature = sm2.sign(message)
    r, s = signature
    print(f"签名结果: r={hex(r)}, s={hex(s)}")
    valid = sm2.verify(message, signature)
    print(f"签名验证结果: {'成功' if valid else '失败'}")

# ========== k值泄露攻击演示 ==========
    print("\n" + "=" * 40)
    print("攻击1: k值泄露攻击演示")
    print("=" * 40)
    k_value = FIXED_K
    signature = sm2.sign(message, k_value)
    r, s = signature
    print(f"使用已知k={hex(k_value)[:20]}... 生成签名: r={hex(r)}, s={hex(s)}")

    # 攻击者使用k值推导私钥
    private_key_guess = sm2.k_leakage_attack(signature, k_value, message, public_key)
    print(f"\n攻击结果 - 推导私钥: ", end = "")
    if private_key_guess is not None:
print(f"{hex(private_key_guess)}")
    else:
print("无法推导私钥（计算错误）")

print(f"真实私钥: {hex(private_key)}")
if private_key_guess is not None and private_key_guess == private_key :
    print(">>> 攻击成功! 推导私钥与真实私钥匹配 <<<")
else:
print(">>> 攻击失败! 推导私钥与真实私钥不匹配 <<<")

# ========== 相同k值重复使用攻击演示 ==========
print("\n" + "=" * 40)
print("攻击2: 相同k值重复使用攻击演示")
print("=" * 40)
message1 = "message digest"
message2 = "message digest1"

# 使用相同k值生成两个签名
k_value = FIXED_K
signature1 = sm2.sign(message1, k_value)
signature2 = sm2.sign(message2, k_value)
r1, s1 = signature1
r2, s2 = signature2

print(f"消息1 '{message1}' 签名: r={hex(r1)}, s={hex(s1)}")
print(f"消息2 '{message2}' 签名: r={hex(r2)}, s={hex(s2)}")
print(f"相同k={hex(k_value)[:20]}... 用于两个签名")

# 攻击者使用两个签名推导私钥
private_key_guess = sm2.repeated_k_attack(signature1, message1, signature2, message2, public_key)
print(f"\n攻击结果 - 推导私钥: ", end = "")
if private_key_guess is not None:
print(f"{hex(private_key_guess)}")
else:
print("无法推导私钥（计算错误）")

print(f"真实私钥: {hex(private_key)}")
if private_key_guess is not None and private_key_guess == private_key :
    print(">>> 攻击成功! 推导私钥与真实私钥匹配 <<<")
else:
print(">>> 攻击失败! 推导私钥与真实私钥不匹配 <<<")

# ========== 多个用户共享相同k值攻击演示 ==========
print("\n" + "=" * 40)
print("攻击3: 多个用户共享相同k值攻击")
print("=" * 40)
# 创建Alice用户
alice_id = "ALICE123@YAHOO.COM"
alice_private_key = private_key
alice_public_key = public_key
print(f"Alice私钥: {hex(alice_private_key)}")

# 创建Bob用户
bob_id = "BOB123@YAHOO.COM"
bob_private_key = random.randint(1, N - 1)
bob_public_key = elliptic_multiply(bob_private_key, G)
print(f"Bob私钥: {hex(bob_private_key)}")

# 使用相同k值生成签名
k_value = FIXED_K
message_alice = "Alice's message"
message_bob = "Bob's message"

# Alice的签名
alice_signature = sm2.sign(message_alice, k_value)

# Bob的签名（创建新的SM2实例以使用Bob的密钥和ID）
sm2_bob = SM2(private_key = bob_private_key, public_key = bob_public_key, user_id = bob_id)
bob_signature = sm2_bob.sign(message_bob, k_value)

print(f"Alice签名: r={hex(alice_signature[0])}, s={hex(alice_signature[1])}")
print(f"Bob签名: r={hex(bob_signature[0])}, s={hex(bob_signature[1])}")

# 攻击者推导Alice的私钥
priv_alice_guess = sm2.multi_user_k_attack(
    alice_signature,
    message_alice,
    alice_public_key,
    alice_id,
    k_value
)

# 攻击者推导Bob的私钥（使用Bob的SM2实例确保正确的用户ID）
priv_bob_guess = sm2_bob.multi_user_k_attack(
    bob_signature,
    message_bob,
    bob_public_key,
    bob_id,
    k_value
)

print(f"\n攻击结果 - Alice私钥推导: ", end = "")
if priv_alice_guess is not None:
print(f"{hex(priv_alice_guess)}")
else:
print("无法推导私钥（计算错误）")

print(f"真实Alice私钥: {hex(alice_private_key)}")
if priv_alice_guess is not None and priv_alice_guess == alice_private_key :
    print(">>> Alice私钥攻击成功! <<<")
else:
print(">>> Alice私钥攻击失败! <<<")

print(f"\n攻击结果 - Bob私钥推导: ", end = "")
if priv_bob_guess is not None :
    print(f"{hex(priv_bob_guess)}")
else:
print("无法推导私钥（计算错误）")

print(f"真实Bob私钥: {hex(bob_private_key)}")
if priv_bob_guess is not None and priv_bob_guess == bob_private_key :
    print(">>> Bob私钥攻击成功! <<<")
else:
print(">>> Bob私钥攻击失败! <<<")

print("\n" + "=" * 80)
print("攻击演示完成".center(80))
print("=" * 80)









import random
from gmssl import sm3, func

# SM2椭圆曲线参数
PRIME = 0x8542D69E4C044F18E8B92435BF6FF7DE457283915C45517D722EDB8B08F1DFC3
A = 0x787968B4FA32C3FD2417842E73BBFEFF2F3C848B6831D7E0EC65228B3937E498
B = 0x63E4C6D3B23B0C849CF84241484BFE48F61D59A5B16BA06E6E12D1DA27C5249A
Gx = 0x421DEBD61B62EAB6746434EBC3CC315E32220B3BADD50BDC4C4E6C147FEDD43D
Gy = 0x0680512BCBB42C07D47349D2153B70C4E5D7FDFCBFA36EA1A85841B9E46E09A2
N = 0x8542D69E4C044F18E8B92435BF6FF7DD297720630485628D5AE74EE7C32E79B7
G = (Gx, Gy)


def mod_inv(a, modulus = N) :
    """扩展欧几里得算法求模逆"""
    if a == 0 :
        return 0
        lm, hm = 1, 0
        low, high = a % modulus, modulus
        while low > 1:
r = high // low
nm, new = hm - lm * r, high - low * r
lm, low, hm, high = nm, new, lm, low
return lm % modulus


def elliptic_add(point1, point2) :
    """椭圆曲线点加法"""
    if point1 == (0, 0) :
        return point2
        if point2 == (0, 0) :
            return point1
            if point1 == point2 :
                return elliptic_double(point1)
                if point1[0] == point2[0] :
                    return (0, 0)  # 无穷远点

                    dx = (point2[0] - point1[0]) % PRIME
                    dy = (point2[1] - point1[1]) % PRIME
                    s = dy * mod_inv(dx, PRIME) % PRIME
                    x = (s * *2 - point1[0] - point2[0]) % PRIME
                    y = (s * (point1[0] - x) - point1[1]) % PRIME
                    return (x, y)


                    def elliptic_double(point) :
                    """椭圆曲线倍点运算"""
                    if point == (0, 0) :
                        return point

                        numerator = (3 * point[0] * *2 + A) % PRIME
                        denominator = (2 * point[1]) % PRIME
                        s = numerator * mod_inv(denominator, PRIME) % PRIME

                        x = (s * *2 - 2 * point[0]) % PRIME
                        y = (s * (point[0] - x) - point[1]) % PRIME
                        return (x, y)


                        def elliptic_multiply(k, point) :
                        """椭圆曲线点乘法（标量乘法）"""
                        if k% N == 0 or point == (0, 0) :
                            return (0, 0)
                            if k < 0 :
                                return elliptic_multiply(-k, point)

                                result = (0, 0)
                                addend = point

                                while k:
if k & 1 :
    result = elliptic_add(result, addend)
    addend = elliptic_double(addend)
    k >>= 1

    return result


    def get_za(user_id, public_key) :
    """计算Z_A值（SM2预计算哈希部分）"""
    if user_id is None :
user_id = "ALICE123@YAHOO.COM"

entl = len(user_id.encode('utf-8')) * 8
entl_bytes = entl.to_bytes(2, 'big')

a_bytes = A.to_bytes(32, 'big')
b_bytes = B.to_bytes(32, 'big')
gx_bytes = Gx.to_bytes(32, 'big')
gy_bytes = Gy.to_bytes(32, 'big')
px_bytes = public_key[0].to_bytes(32, 'big')
py_bytes = public_key[1].to_bytes(32, 'big')

za_input = (
    entl_bytes +
    user_id.encode('utf-8') +
    a_bytes +
    b_bytes +
    gx_bytes +
    gy_bytes +
    px_bytes +
    py_bytes
    )

    return sm3.sm3_hash(func.bytes_to_list(za_input))


    def compute_hash(za, message) :
    """计算消息哈希e = Hash(Z_A || M)"""
    msg_bytes = message.encode('utf-8')
    za_bytes = bytes.fromhex(za)
    input_data = func.bytes_to_list(za_bytes + msg_bytes)
    hash_hex = sm3.sm3_hash(input_data)
    return int(hash_hex, 16) % N


    class SM2 :
    def __init__(self, private_key = None, public_key = None, user_id = None) :
    self.user_id = user_id or "ALICE123@YAHOO.COM"
    self.private_key = private_key
    self.public_key = public_key

    if self.private_key and not self.public_key :
        self.public_key = elliptic_multiply(self.private_key, G)

        if self.public_key :
            self.za = get_za(self.user_id, self.public_key)
        else :
            self.za = None

            def sign(self, message, k = None) :
            """生成SM2签名"""
            if not self.private_key :
                raise ValueError("需要私钥才能签名")

                e = compute_hash(self.za, message)

                if k is None :
k = random.randint(1, N - 1)
                else:
k = k % N

x1, y1 = elliptic_multiply(k, G)
r = (e + x1) % N

if r == 0 or (r + k) % N == 0:
return self.sign(message)

d = self.private_key
s = mod_inv(1 + d, N) * (k - r * d) % N

if s < 0:
s += N

if s == 0 :
    return self.sign(message)

    return (r, s)

    def verify(self, message, signature) :
    """验证SM2签名"""
    if not self.public_key :
        raise ValueError("需要公钥才能验证签名")

        r, s = signature

        if not (1 <= r <= N - 1 and 1 <= s <= N - 1) :
            return False

            e = compute_hash(self.za, message)
            t = (r + s) % N

            if t == 0:
return False

sG = elliptic_multiply(s, G)
tP = elliptic_multiply(t, self.public_key)
x1, y1 = elliptic_add(sG, tP)

R = (e + x1) % N
return R == r


def forge_satoshi_signature(message, satoshi_public_key, user_id = "SATOSHI@BITCOIN.COM") :
    """伪造中本聪的SM2数字签名（修正版）"""
    # 步骤1：计算ZA值
    za = get_za(user_id, satoshi_public_key)

    # 步骤2：计算消息哈希e
    e = compute_hash(za, message)

    # 步骤3：随机选择u和v
    u = random.randint(1, N - 1)
    v = random.randint(1, N - 1)

    # 步骤4：计算伪造点R' = [u]G + [v]P
    uG = elliptic_multiply(u, G)
    vP = elliptic_multiply(v, satoshi_public_key)
    x1, y1 = elliptic_add(uG, vP)

    # 步骤5：构造r = (e + x1) mod N
    r = (e + x1) % N
    if r == 0:
# 如果r为0，重试
return forge_satoshi_signature(message, satoshi_public_key, user_id)

# 步骤6：构造s = [(r / v) - u] mod N
# 注意：这里需要计算 v 的模逆
s = (mod_inv(v, N) * (r - u)) % N
if s == 0:
# 如果s为0，重试
return forge_satoshi_signature(message, satoshi_public_key, user_id)

# 步骤7：调整s值为正数
if s < 0:
s += N

return (r, s)


# 演示代码
if __name__ == "__main__":
print("=== SM2数字签名伪造演示 ===")

# 生成示例中本聪的密钥对（实际中仅公钥公开）
satoshi_private_key = random.randint(1, N - 1)
satoshi_public_key = elliptic_multiply(satoshi_private_key, G)
print(f"中本聪公钥: (x={hex(satoshi_public_key[0])[:16]}..., y={hex(satoshi_public_key[1])[:16]}...)")

# 要签名的消息
message = "I am Satoshi Nakamoto. This message is authentic."
print(f"\n消息: {message}")

# 生成真实签名（作为对比）
satoshi_sm2 = SM2(private_key = satoshi_private_key, public_key = satoshi_public_key,
    user_id = "SATOSHI@BITCOIN.COM")
    real_signature = satoshi_sm2.sign(message)
    r_real, s_real = real_signature
    print(f"\n真实签名: r={hex(r_real)[:16]}..., s={hex(s_real)[:16]}...")
    print(f"真实签名验证: {'有效' if satoshi_sm2.verify(message, real_signature) else '无效'}")

    # 伪造签名（使用修正后的方法）
    print("\n=== 使用修正后的方法伪造签名 ===")
    forged_signature = forge_satoshi_signature(message, satoshi_public_key, "SATOSHI@BITCOIN.COM")

    if forged_signature:
r_forge, s_forge = forged_signature
print(f"伪造签名: r={hex(r_forge)[:16]}..., s={hex(s_forge)[:16]}...")

# 验证伪造的签名
verifier = SM2(public_key = satoshi_public_key, user_id = "SATOSHI@BITCOIN.COM")
is_valid = verifier.verify(message, forged_signature)
print(f"伪造签名验证: {'有效' if is_valid else '无效'}")

if is_valid:
print("成功！SM2签名伪造演示完成")
else:
print("伪造失败，签名无效")
    else:
print("无法生成伪造签名")*/