/*#include <iostream>
#include <string>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>
using namespace std;

// ȷ���ַ�������Ϊָ��ֵ��������0
string padString(const string& s, int length) {
    if (s.size() >= length) return s.substr(0, length);
    return string(length - s.size(), '0') + s;
}

// ������ת��Ϊʮ�����ƺ���
string BinToHex(string str) {
    string hex = "";
    int temp = 0;
    while (str.size() % 4 != 0) {
        str = "0" + str;
    }
    for (size_t i = 0; i < str.size(); i += 4) {
        if (i + 3 >= str.size()) break; // ��ֹԽ��
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

// ʮ������ת��Ϊ�����ƺ���
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
            idx = 0; // ��Ч�ַ�����
        }
        bin += table[idx];
    }
    return bin;
}

// ʮ����ת��Ϊʮ�����ƺ�������ȫָ����ȣ�
string DecToHex(uint64_t num, int width = 8) {
    stringstream ss;
    ss << hex << setw(width) << setfill('0') << num;
    string res = ss.str();
    transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

// �ַ���ת��Ϊʮ�������ַ���
string strToHex(const string& input) {
    stringstream ss;
    ss << hex << setfill('0');
    for (unsigned char c : input) {
        ss << setw(2) << static_cast<int>(c);
    }
    return ss.str();
}

// ѭ������
string LeftShift(string str, int len) {
    str = padString(str, 8); // ȷ��8λʮ������
    string bin = HexToBin(str);
    bin = padString(bin, 32); // ȷ��32λ������
    len = len % 32; // ��ֹ��λ����
    if (len < 0) len += 32; // ������λ
    string shifted = bin.substr(len) + bin.substr(0, len);
    return BinToHex(shifted);
}

// ������
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

// �����
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

// �����
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

// �ǲ���
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

// ģ�����㣨ģ2^32��
string ModAdd(string a, string b) {
    a = padString(a, 8);
    b = padString(b, 8);

    unsigned long long a_val = stoull(a, nullptr, 16);
    unsigned long long b_val = stoull(b, nullptr, 16);
    unsigned long long result = (a_val + b_val) % 0x100000000ULL;

    return DecToHex(static_cast<uint64_t>(result));
}

// �û�����P0
string P0(string str) {
    return XOR(XOR(str, LeftShift(str, 9)), LeftShift(str, 17));
}

// �û�����P1
string P1(string str) {
    return XOR(XOR(str, LeftShift(str, 15)), LeftShift(str, 23));
}

// Tj����
string T(int j) {
    if (j >= 0 && j <= 15) {
        return "79cc4519";
    }
    else {
        return "7a879d8a";
    }
}

// ��������FF
string FF(string X, string Y, string Z, int j) {
    if (j >= 0 && j <= 15) {
        return XOR(XOR(X, Y), Z);
    }
    else {
        return OR(OR(AND(X, Y), AND(X, Z)), AND(Y, Z));
    }
}

// ��������GG
string GG(string X, string Y, string Z, int j) {
    if (j >= 0 && j <= 15) {
        return XOR(XOR(X, Y), Z);
    }
    else {
        return OR(AND(X, Y), AND(NOT(X), Z));
    }
}

// ��Ϣ��չ����
string extension(string str) {
    // ȷ��������128�ַ���512λ��
    if (str.size() < 128) {
        str += string(128 - str.size(), '0');
    }
    else if (str.size() > 128) {
        str = str.substr(0, 128);
    }

    string W = str;
    // ����W16��W67
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

    // ����W'0��W'63
    for (int j = 0; j < 64; ++j) {
        if (j * 8 + 8 > W.size() || (j + 4) * 8 + 8 > W.size()) break;
        string wj = XOR(W.substr(j * 8, 8), W.substr((j + 4) * 8, 8));
        W += wj;
    }

    return W;
}

// ѹ������
string compress(string W, string IV) {
    // ȷ��IV��64�ַ���256λ��
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
        // ȷ��W���㹻�ĳ���
        if ((j + 68) * 8 + 8 > W.size() || j * 8 + 8 > W.size()) break;

        // ����SS1
        string shlA12 = LeftShift(A, 12);
        string t = T(j);
        string shlT = LeftShift(t, j % 32);
        string add1 = ModAdd(shlA12, E);
        string add2 = ModAdd(add1, shlT);
        string SS1 = LeftShift(add2, 7);

        // ����SS2
        string SS2 = XOR(SS1, shlA12);

        // ����TT1
        string ff = FF(A, B, C, j);
        string add3 = ModAdd(ff, D);
        string add4 = ModAdd(add3, SS2);
        string wp = W.substr((j + 68) * 8, 8); // W'j
        string TT1 = ModAdd(add4, wp);

        // ����TT2
        string gg = GG(E, F, G, j);
        string add5 = ModAdd(gg, H);
        string add6 = ModAdd(add5, SS1);
        string wj = W.substr(j * 8, 8); // Wj
        string TT2 = ModAdd(add6, wj);

        // ���¼Ĵ���
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

// ��Ϣ��亯��
string padding(const string& msg_hex, uint64_t& total_bit_len) {
    int msg_bit_len = msg_hex.size() * 4;
    total_bit_len = msg_bit_len;

    // ������1 + k��0 + 64λ����
    int k = (448 - (msg_bit_len + 1) % 512 + 512) % 512;
    string padding_str;

    // ���'1'λ��k��'0'λ��ת��Ϊʮ�����ƣ�
    padding_str += "8"; // ������1000
    padding_str += string(k / 4, '0'); // ÿ��ʮ�������ַ�4λ

    // ���64λ���ȣ������
    padding_str += DecToHex(total_bit_len, 16);

    return msg_hex + padding_str;
}

// SM3��ϣ����
string sm3_hash(const string& msg_hex, string IV = "7380166f4914b2b9172442d7da8a0600a96f30bc163138aae38dee4db0fb0e4e") {
    uint64_t total_len;
    string padded = padding(msg_hex, total_len);
    int num_blocks = padded.size() / 128;
    if (padded.size() % 128 != 0) num_blocks++; // �������һ����������

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

// ������չ��������
string length_extension_attack(const string& original_hash, int original_bit_len, const string& extension) {
    // 1. ����ԭʼ��Ϣ����䳤��
    int k = (448 - (original_bit_len + 1) % 512 + 512) % 512;
    int pad_bit_len = 1 + k + 64;

    // 2. ��չ��Ϣ����
    string ext_hex = strToHex(extension);
    uint64_t ext_total_len = original_bit_len + pad_bit_len + ext_hex.size() * 4;

    // 3. ������չ��Ϣ�����
    string ext_padded = ext_hex;
    uint64_t dummy;
    ext_padded = padding(ext_padded, dummy);

    // 4. ʹ��ԭʼ��ϣ��Ϊ��ʼ��������
    return sm3_hash(ext_padded, original_hash);
}

int main() {
    try {
        cout << "SM3������չ������֤" << endl;
        cout << "==========================================" << endl;

        // ԭʼ��Ϣ
        string original_msg = "Hello, world!";
        int original_bit_len = original_msg.size() * 8;

        cout << "ԭʼ��Ϣ: \"" << original_msg << "\"" << endl;
        cout << "ԭʼ��Ϣ����: " << original_bit_len << " ����" << endl;

        // ����ԭʼ��ϣ
        string original_hex = strToHex(original_msg);
        string original_hash = sm3_hash(original_hex);
        cout << "ԭʼ��ϣֵ: " << original_hash << endl << endl;

        // ��չ��Ϣ
        string extension_msg = " This is an extension!";
        cout << "��չ��Ϣ: \"" << extension_msg << "\"" << endl;

        // ִ�г�����չ����
        string forged_hash = length_extension_attack(original_hash, original_bit_len, extension_msg);
        cout << "�������ɵ�α���ϣֵ: " << forged_hash << endl << endl;

        // ������ʵ������Ϣ��ϣ��ԭʼ��Ϣ + ��� + ��չ��Ϣ��
        string original_padded_hex;
        uint64_t dummy_len;
        original_padded_hex = padding(original_hex, dummy_len);
        string new_msg_hex = original_padded_hex + strToHex(extension_msg);
        string real_hash = sm3_hash(new_msg_hex);
        cout << "��ʵ������Ϣ��ϣֵ: " << real_hash << endl << endl;

        // ��֤���
        if (forged_hash == real_hash) {
            cout << "��֤���: ������չ�����ɹ�!" << endl;
            cout << "��������δ֪ԭʼ��Ϣ������£��ɹ�α������չ��Ϣ�Ĺ�ϣֵ" << endl;
        }
        else {
            cout << "��֤���: ������չ����ʧ��!" << endl;
        }
    }
    catch (const exception& e) {
        cerr << "����ʱ����: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        cerr << "����δ֪����" << endl;
        return 1;
    }

    return 0;
}



import random
import hashlib
import math

# ģ���̬ͬ����ϵͳ��������ʾ��
class SimpleHomomorphicEncryption :
    def __init__(self) :
    # ���ɴ����� p �� q
    self.p = self.generate_large_prime(1024)
    self.q = self.generate_large_prime(1024)
    self.n = self.p * self.q
    self.nsquare = self.n * self.n

    def generate_large_prime(self, bits) :
    """���ɴ���������ʵ�֣�"""
    while True :
        p = random.randint(2 * *(bits - 1), 2 * *bits)
        if p % 2 != 0 and self.is_prime(p) :
            return p

            def is_prime(self, n, k = 10) :
            """����-������������"""
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
                                            """ģ�����"""
                                            r = random.randint(1, self.n - 1)
                                            # Paillier ���ܹ�ʽ�ļ�ģ�� : (1 + plaintext * n) * r ^ n mod n ^ 2
                                            ciphertext = (1 + plaintext * self.n) * pow(r, self.n, self.nsquare) % self.nsquare
                                            return { 'ciphertext': ciphertext, 'r' : r }

                                            def decrypt(self, ciphertext) :
                                            """ģ�����"""
                                            # �򻯽��ܹ���
                                            return (ciphertext - 1) // self.n

                                            def raw_decrypt(self, ciphertext) :
                                            """ֱ�ӽ�����������"""
                                            return (ciphertext - 1) // self.n

                                            # ʹ�ü򻯵Ĺ�ϣ����
                                            def hash_to_point(u: str)->bytes:
"""���ַ���ӳ��Ϊ�̶����ȵĹ�ϣֵ"""
return hashlib.sha256(u.encode()).digest()

# �򻯵ĵ�����
def point_operation(value: bytes, key : int)->bytes:
"""ʹ����Կ�Թ�ϣֵ���д���"""
key_bytes = key.to_bytes(32, 'big')
# ʹ���������͹�ϣ���
transformed = bytes([v ^ k for v, k in zip(value, key_bytes)])
return hashlib.sha256(transformed).digest()

class Party1 :
    def __init__(self, set_V: list) :
    self.set_V = set_V
    self.k1 = random.randint(1, 2 * *256 - 1)  # 256λ�����Կ
    self.paillier_pubkey = None
    self.intersection_sum_cipher = None

    def set_paillier_pubkey(self, pubkey) :
    self.paillier_pubkey = pubkey

    def round1(self)->list :
    # ���� H(v_i) ^ k1
    self.hashed_exponents = []
    for v in self.set_V:
P = hash_to_point(v)
Q = point_operation(P, self.k1)
self.hashed_exponents.append(Q)

# ���ϴ��
random.shuffle(self.hashed_exponents)
return self.hashed_exponents

def round3(self, Z: list, received_set : list) :
    # ��Z����ת��Ϊ�ֽڴ�����
    Z_bytes_set = set(Z)

    # ������յ�Ԫ��[(H(w_j) ^ k2, enc(t_j)]
        intersection_ciphers = []
        for point_bytes, enc_t in received_set :
# ���� H(w_j)^ { k1 * k2 }
T = point_operation(point_bytes, self.k1)

# ����Ƿ���Z��
if T in Z_bytes_set :
intersection_ciphers.append(enc_t)

# ̬ͬ���
if intersection_ciphers:
product = 1
nsquare = self.paillier_pubkey.nsquare
# �������������
for c in intersection_ciphers :
product = (product * c) % nsquare

# ˢ������(���������)
r = random.randint(1, self.paillier_pubkey.n - 1)
c0 = pow(r, self.paillier_pubkey.n, nsquare)  # ����0
refreshed_cipher = (product * c0) % nsquare
else:
# �������Ϊ�գ�����0
refreshed_cipher = self.paillier_pubkey.encrypt(0)['ciphertext']

self.intersection_sum_cipher = refreshed_cipher
return refreshed_cipher

class Party2 :
    def __init__(self, set_W: list) :
    self.set_W = set_W
    self.k2 = random.randint(1, 2 * *256 - 1)  # 256λ�����Կ
    # ����̬ͬ������Կ
    self.paillier_public_key = SimpleHomomorphicEncryption()
    self.paillier_private_key = self.paillier_public_key
    self.intersection_sum = None

    def get_paillier_pubkey(self) :
    return self.paillier_public_key

    def round2(self, received_points: list)->tuple :
    # ����1 : ���� H(v_i)^ { k1 * k2 }
Z = []
for P in received_points :
Q = point_operation(P, self.k2)
Z.append(Q)
random.shuffle(Z)

# ����2 : �����Լ��ļ���
processed_set = []
for w, t in self.set_W :
    # ���� H(w_j)^ { k2 }
P = hash_to_point(w)
Q = point_operation(P, self.k2)
# ���� t_j
enc_t = self.paillier_public_key.encrypt(t)['ciphertext']
processed_set.append((Q, enc_t))

random.shuffle(processed_set)
return Z, processed_set

def decrypt_sum(self, ciphertext: int) -> int:
# ֱ�ӽ�����������
self.intersection_sum = self.paillier_private_key.raw_decrypt(ciphertext)
return self.intersection_sum

# ����Э��ִ��
if __name__ == "__main__":
# ʾ�����ݼ�
set_V = ["id1", "id2", "id3", "id5"]
set_W = [("id1", 10), ("id2", 20), ("id3", 30), ("id4", 40)]

# ��ʼ�����뷽
p1 = Party1(set_V)
p2 = Party2(set_W)

# ���ý׶�: P2���͹�Կ��P1
p1.set_paillier_pubkey(p2.get_paillier_pubkey())

# ��1�� : P1->P2
r1_output = p1.round1()

# ��2�� : P2->P1(����Z�ʹ����ļ���)
Z, r2_output = p2.round2(r1_output)

# ��3�� : P1->P2(���ͼ��ܵĺ�)
sum_cipher = p1.round3(Z, r2_output)

# P2���ܵõ����
result = p2.decrypt_sum(sum_cipher)

print(f"����Ԫ��: id1, id2, id3 (Ԥ�ں�Ϊ60)")
print(f"ʵ�ʼ���õ��Ľ�����: {result}")
















import os
import random
import binascii
from gmssl import sm3, func

# ===== ����ģʽ���� =====
DEBUG_FIXED_KEYS = True  # �̶�˽Կ / ��Կ
DEBUG_FIXED_K = True  # �̶�ǩ������� k
FIXED_PRIVATE_KEY = int(
    "128D97B99C874D5443E4D2F2A9FA9130EBF2B7E9E1E5D7A441BDCE3D1F2A29AC", 16
    )
    FIXED_K = int(
        "1F1E1D1C1B1A191817161514131211100F0E0D0C0B0A09080706050403020100", 16
        )
# ======================

    # ��Բ���߲�����SM2�Ƽ�������
    PRIME = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
    A = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
    B = 0x28E9FA9E9D9F5E344D5AEF7F6BFFFF5F
    Gx = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
    Gy = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
    N = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
    G = (Gx, Gy)


    def mod_inv(a, modulus = PRIME) :
    """��չŷ������㷨��ģ��"""
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
    """��Բ���ߵ�ӷ�"""
    if point1[0] == point2[0] and point1[1] == point2[1] :
        return elliptic_double(point1)
        if point1 == (0, 0) :
            return point2
            if point2 == (0, 0) :
                return point1
                if point1[0] == point2[0] :
                    return (0, 0)

                    # ����б��
                    dx = (point2[0] - point1[0]) % PRIME
                    dy = (point2[1] - point1[1]) % PRIME
                    s = dy * mod_inv(dx, PRIME)
                    x = (s * *2 - point1[0] - point2[0]) % PRIME
                    y = (s * (point1[0] - x) - point1[1]) % PRIME
                    return (x, y)


                    def elliptic_double(point) :
                    """��Բ���߱�������"""
                    if point == (0, 0) :
                        return point

                        # ����б��
                        numerator = (3 * point[0] * *2 + A) % PRIME
                        denominator = (2 * point[1]) % PRIME
                        s = numerator * mod_inv(denominator, PRIME)

                        x = (s * *2 - 2 * point[0]) % PRIME
                        y = (s * (point[0] - x) - point[1]) % PRIME
                        return (x, y)


                        def elliptic_multiply(k, point) :
                        """��Բ���ߵ�˷��������˷���"""
                        if k% N == 0 or point == (0, 0) :
                            return (0, 0)
                            if k < 0 :
                                return elliptic_multiply(-k, point)

                                # ������չ��
                                result = (0, 0)
                                addend = point

                                while k:
if k & 1 :
    result = elliptic_add(result, addend)
    addend = elliptic_double(addend)
    k >>= 1

    return result


    def get_za(user_id, public_key) :
    """����Z_Aֵ��SM2Ԥ�����ϣ���֣�"""
    # �û�IDĬ��ֵ
    if user_id is None :
user_id = "1234567812345678"

entl = len(user_id.encode('utf-8')) * 8
entl_bytes = entl.to_bytes(2, 'big')

# ת�����߲������ֽ�
a_bytes = A.to_bytes(32, 'big')
b_bytes = B.to_bytes(32, 'big')
gx_bytes = Gx.to_bytes(32, 'big')
gy_bytes = Gy.to_bytes(32, 'big')

# ת����Կ���ֽ�
px_bytes = public_key[0].to_bytes(32, 'big')
py_bytes = public_key[1].to_bytes(32, 'big')

# ����Z_A��ϣ����
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

    # �����ϣ������ʮ�������ַ���
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

    # ���㲢�洢ZA
    self.za = get_za(self.user_id, self.public_key) if self.public_key else None

    def generate_key_pair(self) :
    """������Կ��"""
    self.private_key = random.randint(1, N - 1)
    self.public_key = elliptic_multiply(self.private_key, G)
    self.za = get_za(self.user_id, self.public_key)
    return self.private_key, self.public_key

    def sign(self, message) :
    """SM2ǩ��"""
    if not self.private_key :
        raise ValueError("Private key is required for signing")

        e = self._hash_message(message)
        if DEBUG_FIXED_K :
            k = FIXED_K % N
        else:
k = random.randint(1, N - 1)

# ������Բ���ߵ�[k]G
x1, y1 = elliptic_multiply(k, G)
r = (e + x1) % N

# ����rΪ0��r + k == n�����
if r == 0 or r + k == N:
return self.sign(message)

# ����s = (1 + d)^ { -1 } *(k - r * d) mod n
d = self.private_key
s = (pow(1 + d, N - 2, N) * (k - r * d)) % N

# ����sΪ0
if s == 0:
return self.sign(message)

return (r, s)

def verify(self, message, signature) :
    """SM2��֤ǩ��"""
    if not self.public_key :
        raise ValueError("Public key is required for verification")

        r, s = signature

        # ���r��s�Ƿ���[1, n - 1]��Χ��
        if not (1 <= r <= N - 1 and 1 <= s <= N - 1) :
            return False

            e = self._hash_message(message)
            t = (r + s) % N

            # ������Բ���ߵ�[s]G + [t]P
            point1 = elliptic_multiply(s, G)
            point2 = elliptic_multiply(t, self.public_key)
            x1, y1 = elliptic_add(point1, point2)

            # ��֤R = (e + x1) mod n
            return (r % N) == ((e + x1) % N)

            def _hash_message(self, message) :
            """������Ϣ��ϣe = Hash(Z_A || M)"""
            if not self.za :
                raise ValueError("ZA not initialized")

                msg_bytes = message.encode('utf-8')
                # ��ZA��ʮ�������ַ�����ת��Ϊ�ֽ�����
                za_bytes = bytes.fromhex(self.za)
                input_data = func.bytes_to_list(za_bytes + msg_bytes)
                hash_hex = sm3.sm3_hash(input_data)
                return int(hash_hex, 16)


# ======== �������� ========
                if __name__ == "__main__":
# 1. ��Կ���ɲ���
print("===== SM2�㷨ʵ�ֲ��� =====")
sm2 = SM2(user_id = "test@example.com")
print("��Կ���ɳɹ�:", (hex(sm2.public_key[0])[:16] + "...", hex(sm2.public_key[1])[:16] + "..."))

# 2. ǩ������
message = "Hello, SM2!"
signature = sm2.sign(message)
print(f"\n��Ϣ '{message}' ��ǩ��:")
print(f"r: {hex(signature[0])}")
print(f"s: {hex(signature[1])}")

# 3. ��֤����
verify_result = sm2.verify(message, signature)
print(f"\nǩ����֤���: {'�ɹ�' if verify_result else 'ʧ��'}")

# 4. �۸���Ϣ��֤
tampered_message = "Hello, SM3!"
verify_result_tampered = sm2.verify(tampered_message, signature)
print(f"�۸���Ϣ��֤���: {'�ɹ�' if verify_result_tampered else 'ʧ��'}")

# 5. ʹ�ò�ͬ�û�ID����
print("\n===== ��ͬ�û�ID���� =====")
sm2_custom = SM2(user_id = "Alice@email.com")
signature_custom = sm2_custom.sign(message)
verify_result_custom = sm2_custom.verify(message, signature_custom)
print(f"�Զ���IDǩ����֤: {'�ɹ�' if verify_result_custom else 'ʧ��'}")

# ��֤��ͬ�û�ID��ǩ���Ƿ���ͬ
same_signature = signature == signature_custom
print(f"��ͬ�û�ID��ǩ���Ƿ���ͬ: {'��' if same_signature else '��'}")

print("\n���в������!")

























import random
from math import gcd, ceil, log
from gmssl import sm3, func
import binascii

# ===== ����ģʽ���� =====
DEBUG_FIXED_KEYS = True  # �̶�˽Կ / ��Կ
DEBUG_FIXED_K = True  # �̶�ǩ������� k
FIXED_PRIVATE_KEY = int(
    "128B2FA8BD433C6C068C8D803DFF79792A519A55171B1B650C23661D15897263", 16
    )
    FIXED_K = int(
        "6CB28D99385C175C94F94E934817663FC176D925DD72B727260DBAAE1FB2F96F", 16
        )
# ======================

    # ��Բ���߲�������ɹ���������һ�£�
    PRIME = 0x8542D69E4C044F18E8B92435BF6FF7DE457283915C45517D722EDB8B08F1DFC3
    A = 0x787968B4FA32C3FD2417842E73BBFEFF2F3C848B6831D7E0EC65228B3937E498
    B = 0x63E4C6D3B23B0C849CF84241484BFE48F61D59A5B16BA06E6E12D1DA27C5249A
    Gx = 0x421DEBD61B62EAB6746434EBC3CC315E32220B3BADD50BDC4C4E6C147FEDD43D
    Gy = 0x0680512BCBB42C07D47349D2153B70C4E5D7FDFCBFA36EA1A85841B9E46E09A2
    N = 0x8542D69E4C044F18E8B92435BF6FF7DD297720630485628D5AE74EE7C32E79B7
    G = (Gx, Gy)


    def mod_inv(a, modulus = N) :
    """��չŷ������㷨��ģ�棬ʹ��N��ΪĬ��ģ��"""
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
    """��Բ���ߵ�ӷ�"""
    if point1 == (0, 0) :
        return point2
        if point2 == (0, 0) :
            return point1
            if point1 == point2 :
                return elliptic_double(point1)
                if point1[0] == point2[0] :
                    return (0, 0)  # ����Զ��

                    # ����б��
                    dx = (point2[0] - point1[0]) % PRIME
                    dy = (point2[1] - point1[1]) % PRIME
                    s = dy * mod_inv(dx, PRIME) % PRIME
                    x = (s * *2 - point1[0] - point2[0]) % PRIME
                    y = (s * (point1[0] - x) - point1[1]) % PRIME
                    return (x, y)


                    def elliptic_double(point) :
                    """��Բ���߱�������"""
                    if point == (0, 0) :
                        return point

                        # ����б��
                        numerator = (3 * point[0] * *2 + A) % PRIME
                        denominator = (2 * point[1]) % PRIME
                        s = numerator * mod_inv(denominator, PRIME) % PRIME

                        x = (s * *2 - 2 * point[0]) % PRIME
                        y = (s * (point[0] - x) - point[1]) % PRIME
                        return (x, y)


                        def elliptic_multiply(k, point) :
                        """��Բ���ߵ�˷��������˷���"""
                        if k% N == 0 or point == (0, 0) :
                            return (0, 0)
                            if k < 0 :
                                return elliptic_multiply(-k, point)

                                # ������չ����
                                result = (0, 0)
                                addend = point

                                while k:
if k & 1 :
    result = elliptic_add(result, addend)
    addend = elliptic_double(addend)
    k >>= 1

    return result


    def get_za(user_id, public_key) :
    """����Z_Aֵ��SM2Ԥ�����ϣ���֣�"""
    # �û�IDĬ��ֵ
    if user_id is None :
user_id = "ALICE123@YAHOO.COM"

entl = len(user_id.encode('utf-8')) * 8
entl_bytes = entl.to_bytes(2, 'big')

# ת�����߲������ֽ�
a_bytes = A.to_bytes(32, 'big')
b_bytes = B.to_bytes(32, 'big')
gx_bytes = Gx.to_bytes(32, 'big')
gy_bytes = Gy.to_bytes(32, 'big')

# ת����Կ���ֽ�
px_bytes = public_key[0].to_bytes(32, 'big')
py_bytes = public_key[1].to_bytes(32, 'big')

# ����Z_A��ϣ����
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

    # �����ϣ������ʮ�������ַ���
    return sm3.sm3_hash(func.bytes_to_list(za_input))


    def compute_hash(za, message) :
    """������Ϣ��ϣe = Hash(Z_A || M)"""
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

    # ���㲢�洢ZA
    if self.public_key:
self.za = get_za(self.user_id, self.public_key)
    else:
self.za = None

def generate_key_pair(self) :
    """������Կ��"""
    self.private_key = random.randint(1, N - 1)
    self.public_key = elliptic_multiply(self.private_key, G)
    self.za = get_za(self.user_id, self.public_key)
    return self.private_key, self.public_key

    def sign(self, message, k = None) :
    """SM2ǩ��"""
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

# ������Բ���ߵ�[k]G
x1, y1 = elliptic_multiply(k, G)
r = (e + x1) % N

# ����rΪ0��r + k == n�����
if r == 0 or (r + k) % N == 0:
return self.sign(message, k)

# ����s = (1 + d)^ { -1 } *(k - r * d) mod n
d = self.private_key
s = mod_inv(1 + d, N) * (k - r * d) % N

# ȷ��sΪ����
if s < 0:
s += N

# ����sΪ0
if s == 0:
return self.sign(message, k)

return (r, s)

def verify(self, message, signature, public_key = None, user_id = None) :
    """SM2��֤ǩ��"""
    if public_key is None :
public_key = self.public_key
if public_key is None :
raise ValueError("Public key is required for verification")

user_id = user_id or self.user_id
za = get_za(user_id, public_key)

r, s = signature

# ���r��s�Ƿ���[1, n - 1]��Χ��
if not (1 <= r <= N - 1 and 1 <= s <= N - 1) :
    return False

    e = compute_hash(za, message)
    t = (r + s) % N
    if t == 0 :
        return False

        # ������Բ���ߵ�[s]G + [t]P
        sG = elliptic_multiply(s, G)
        tP = elliptic_multiply(t, public_key)
        x1, y1 = elliptic_add(sG, tP)

        # ��֤R = (e + x1) mod n
        R = (e + x1) % N
        return R == r

# ===== ������ʾ���� =====
        def k_leakage_attack(self, signature, k, message, public_key = None, user_id = None) :
        """��֪kֵ�Ƶ�˽Կ�Ĺ�����ʾ"""
        if public_key is None :
public_key = self.public_key
if public_key is None :
raise ValueError("Public key required for attack")

user_id = user_id or self.user_id
za = get_za(user_id, public_key)

r, s = signature
e = compute_hash(za, message)

# ������ʽ��d = (k - s) * inv(s + r) mod n
numerator = (k - s) % N
denominator = (s + r) % N

# ����ĸ�Ƿ�Ϊ0
if denominator == 0:
print("���󣺼���˽Կʱ�������ĸ���޷�����ģ��")
return None

inv_denom = mod_inv(denominator)
if inv_denom is None :
return None

private_key_guess = (numerator * inv_denom) % N
# ȷ�����Ϊ����
if private_key_guess < 0:
private_key_guess += N
return private_key_guess

def repeated_k_attack(self, signature1, message1, signature2, message2, public_key = None, user_id = None) :
    """��ͬkֵ�Զ���Ϣǩ���Ĺ�����ʾ - �޸�kֵ�Ƶ���ʽ"""
    if public_key is None :
public_key = self.public_key
if public_key is None :
raise ValueError("Public key required for attack")

user_id = user_id or self.user_id
za = get_za(user_id, public_key)

r1, s1 = signature1
r2, s2 = signature2

# ����������Ϣ�Ĺ�ϣֵ
e1 = compute_hash(za, message1)
e2 = compute_hash(za, message2)

# �����Ƶ�kֵ���㹫ʽ��k = (e1 - e2 + s1 * r1 - s2 * r2) * inv(s2 - s1) mod n
numerator_k = (e1 - e2 + s1 * r1 - s2 * r2) % N
denominator_k = (s2 - s1) % N

# ����ĸ�Ƿ�Ϊ0
if denominator_k == 0:
print("����delta_sΪ0���޷�����ģ��")
return None

inv_denominator_k = mod_inv(denominator_k)
if inv_denominator_k is None :
return None

k_guess = (numerator_k * inv_denominator_k) % N

# �Ƶ�˽Կ
private_key_guess = self.k_leakage_attack(signature1, k_guess, message1, public_key, user_id)
return private_key_guess

def multi_user_k_attack(self, signature, message, public_key, user_id, k) :
    """����û�ʹ����ͬkֵ�Ĺ�����ʾ - �޸�˽Կ�Ƶ���ʽ"""
    # ����Z_A
    za = get_za(user_id, public_key)

    r, s = signature
    e = compute_hash(za, message)

    # ��ȷ��˽Կ�Ƶ���ʽ: d = (k - s * (1 + d)) / r mod n
    # ��������Ϊ : d = (k - s) * inv(r + s) mod n
    numerator = (k - s) % N
    denominator = (r + s) % N

    # ȷ����ĸΪ����
    if denominator < 0:
denominator += N

# ����ĸ�Ƿ�Ϊ0
if denominator == 0:
print("���󣺼���˽Կʱ�������ĸ���޷�����ģ��")
return None

inv_denom = mod_inv(denominator)
if inv_denom is None :
return None

private_key_guess = (numerator * inv_denom) % N
# ȷ�����Ϊ����
if private_key_guess < 0:
private_key_guess += N
return private_key_guess


if __name__ == "__main__" :
    print("SM2����ǩ���㷨�빥����ʾ".center(80, '='))

    # ����SM2ʵ��
    sm2 = SM2()

    # ������Կ��
    print("\n������Կ��...")
    private_key = FIXED_PRIVATE_KEY
    public_key = elliptic_multiply(private_key, G)
    print(f"˽Կ: {hex(private_key)}")
    print(f"��Կ: (x={hex(public_key[0])[:20]}..., y={hex(public_key[1])[:20]}...)")

    # ԭʼ��Ϣ
    message = "message digest"
    print(f"\nԭʼ��Ϣ: '{message}'")

    # ��֤ǩ������
    print("\n===== ��֤ǩ������ =====")
    signature = sm2.sign(message)
    r, s = signature
    print(f"ǩ�����: r={hex(r)}, s={hex(s)}")
    valid = sm2.verify(message, signature)
    print(f"ǩ����֤���: {'�ɹ�' if valid else 'ʧ��'}")

# ========== kֵй¶������ʾ ==========
    print("\n" + "=" * 40)
    print("����1: kֵй¶������ʾ")
    print("=" * 40)
    k_value = FIXED_K
    signature = sm2.sign(message, k_value)
    r, s = signature
    print(f"ʹ����֪k={hex(k_value)[:20]}... ����ǩ��: r={hex(r)}, s={hex(s)}")

    # ������ʹ��kֵ�Ƶ�˽Կ
    private_key_guess = sm2.k_leakage_attack(signature, k_value, message, public_key)
    print(f"\n������� - �Ƶ�˽Կ: ", end = "")
    if private_key_guess is not None:
print(f"{hex(private_key_guess)}")
    else:
print("�޷��Ƶ�˽Կ���������")

print(f"��ʵ˽Կ: {hex(private_key)}")
if private_key_guess is not None and private_key_guess == private_key :
    print(">>> �����ɹ�! �Ƶ�˽Կ����ʵ˽Կƥ�� <<<")
else:
print(">>> ����ʧ��! �Ƶ�˽Կ����ʵ˽Կ��ƥ�� <<<")

# ========== ��ͬkֵ�ظ�ʹ�ù�����ʾ ==========
print("\n" + "=" * 40)
print("����2: ��ͬkֵ�ظ�ʹ�ù�����ʾ")
print("=" * 40)
message1 = "message digest"
message2 = "message digest1"

# ʹ����ͬkֵ��������ǩ��
k_value = FIXED_K
signature1 = sm2.sign(message1, k_value)
signature2 = sm2.sign(message2, k_value)
r1, s1 = signature1
r2, s2 = signature2

print(f"��Ϣ1 '{message1}' ǩ��: r={hex(r1)}, s={hex(s1)}")
print(f"��Ϣ2 '{message2}' ǩ��: r={hex(r2)}, s={hex(s2)}")
print(f"��ͬk={hex(k_value)[:20]}... ��������ǩ��")

# ������ʹ������ǩ���Ƶ�˽Կ
private_key_guess = sm2.repeated_k_attack(signature1, message1, signature2, message2, public_key)
print(f"\n������� - �Ƶ�˽Կ: ", end = "")
if private_key_guess is not None:
print(f"{hex(private_key_guess)}")
else:
print("�޷��Ƶ�˽Կ���������")

print(f"��ʵ˽Կ: {hex(private_key)}")
if private_key_guess is not None and private_key_guess == private_key :
    print(">>> �����ɹ�! �Ƶ�˽Կ����ʵ˽Կƥ�� <<<")
else:
print(">>> ����ʧ��! �Ƶ�˽Կ����ʵ˽Կ��ƥ�� <<<")

# ========== ����û�������ͬkֵ������ʾ ==========
print("\n" + "=" * 40)
print("����3: ����û�������ͬkֵ����")
print("=" * 40)
# ����Alice�û�
alice_id = "ALICE123@YAHOO.COM"
alice_private_key = private_key
alice_public_key = public_key
print(f"Alice˽Կ: {hex(alice_private_key)}")

# ����Bob�û�
bob_id = "BOB123@YAHOO.COM"
bob_private_key = random.randint(1, N - 1)
bob_public_key = elliptic_multiply(bob_private_key, G)
print(f"Bob˽Կ: {hex(bob_private_key)}")

# ʹ����ͬkֵ����ǩ��
k_value = FIXED_K
message_alice = "Alice's message"
message_bob = "Bob's message"

# Alice��ǩ��
alice_signature = sm2.sign(message_alice, k_value)

# Bob��ǩ���������µ�SM2ʵ����ʹ��Bob����Կ��ID��
sm2_bob = SM2(private_key = bob_private_key, public_key = bob_public_key, user_id = bob_id)
bob_signature = sm2_bob.sign(message_bob, k_value)

print(f"Aliceǩ��: r={hex(alice_signature[0])}, s={hex(alice_signature[1])}")
print(f"Bobǩ��: r={hex(bob_signature[0])}, s={hex(bob_signature[1])}")

# �������Ƶ�Alice��˽Կ
priv_alice_guess = sm2.multi_user_k_attack(
    alice_signature,
    message_alice,
    alice_public_key,
    alice_id,
    k_value
)

# �������Ƶ�Bob��˽Կ��ʹ��Bob��SM2ʵ��ȷ����ȷ���û�ID��
priv_bob_guess = sm2_bob.multi_user_k_attack(
    bob_signature,
    message_bob,
    bob_public_key,
    bob_id,
    k_value
)

print(f"\n������� - Alice˽Կ�Ƶ�: ", end = "")
if priv_alice_guess is not None:
print(f"{hex(priv_alice_guess)}")
else:
print("�޷��Ƶ�˽Կ���������")

print(f"��ʵAlice˽Կ: {hex(alice_private_key)}")
if priv_alice_guess is not None and priv_alice_guess == alice_private_key :
    print(">>> Alice˽Կ�����ɹ�! <<<")
else:
print(">>> Alice˽Կ����ʧ��! <<<")

print(f"\n������� - Bob˽Կ�Ƶ�: ", end = "")
if priv_bob_guess is not None :
    print(f"{hex(priv_bob_guess)}")
else:
print("�޷��Ƶ�˽Կ���������")

print(f"��ʵBob˽Կ: {hex(bob_private_key)}")
if priv_bob_guess is not None and priv_bob_guess == bob_private_key :
    print(">>> Bob˽Կ�����ɹ�! <<<")
else:
print(">>> Bob˽Կ����ʧ��! <<<")

print("\n" + "=" * 80)
print("������ʾ���".center(80))
print("=" * 80)









import random
from gmssl import sm3, func

# SM2��Բ���߲���
PRIME = 0x8542D69E4C044F18E8B92435BF6FF7DE457283915C45517D722EDB8B08F1DFC3
A = 0x787968B4FA32C3FD2417842E73BBFEFF2F3C848B6831D7E0EC65228B3937E498
B = 0x63E4C6D3B23B0C849CF84241484BFE48F61D59A5B16BA06E6E12D1DA27C5249A
Gx = 0x421DEBD61B62EAB6746434EBC3CC315E32220B3BADD50BDC4C4E6C147FEDD43D
Gy = 0x0680512BCBB42C07D47349D2153B70C4E5D7FDFCBFA36EA1A85841B9E46E09A2
N = 0x8542D69E4C044F18E8B92435BF6FF7DD297720630485628D5AE74EE7C32E79B7
G = (Gx, Gy)


def mod_inv(a, modulus = N) :
    """��չŷ������㷨��ģ��"""
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
    """��Բ���ߵ�ӷ�"""
    if point1 == (0, 0) :
        return point2
        if point2 == (0, 0) :
            return point1
            if point1 == point2 :
                return elliptic_double(point1)
                if point1[0] == point2[0] :
                    return (0, 0)  # ����Զ��

                    dx = (point2[0] - point1[0]) % PRIME
                    dy = (point2[1] - point1[1]) % PRIME
                    s = dy * mod_inv(dx, PRIME) % PRIME
                    x = (s * *2 - point1[0] - point2[0]) % PRIME
                    y = (s * (point1[0] - x) - point1[1]) % PRIME
                    return (x, y)


                    def elliptic_double(point) :
                    """��Բ���߱�������"""
                    if point == (0, 0) :
                        return point

                        numerator = (3 * point[0] * *2 + A) % PRIME
                        denominator = (2 * point[1]) % PRIME
                        s = numerator * mod_inv(denominator, PRIME) % PRIME

                        x = (s * *2 - 2 * point[0]) % PRIME
                        y = (s * (point[0] - x) - point[1]) % PRIME
                        return (x, y)


                        def elliptic_multiply(k, point) :
                        """��Բ���ߵ�˷��������˷���"""
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
    """����Z_Aֵ��SM2Ԥ�����ϣ���֣�"""
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
    """������Ϣ��ϣe = Hash(Z_A || M)"""
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
            """����SM2ǩ��"""
            if not self.private_key :
                raise ValueError("��Ҫ˽Կ����ǩ��")

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
    """��֤SM2ǩ��"""
    if not self.public_key :
        raise ValueError("��Ҫ��Կ������֤ǩ��")

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
    """α���б��ϵ�SM2����ǩ���������棩"""
    # ����1������ZAֵ
    za = get_za(user_id, satoshi_public_key)

    # ����2��������Ϣ��ϣe
    e = compute_hash(za, message)

    # ����3�����ѡ��u��v
    u = random.randint(1, N - 1)
    v = random.randint(1, N - 1)

    # ����4������α���R' = [u]G + [v]P
    uG = elliptic_multiply(u, G)
    vP = elliptic_multiply(v, satoshi_public_key)
    x1, y1 = elliptic_add(uG, vP)

    # ����5������r = (e + x1) mod N
    r = (e + x1) % N
    if r == 0:
# ���rΪ0������
return forge_satoshi_signature(message, satoshi_public_key, user_id)

# ����6������s = [(r / v) - u] mod N
# ע�⣺������Ҫ���� v ��ģ��
s = (mod_inv(v, N) * (r - u)) % N
if s == 0:
# ���sΪ0������
return forge_satoshi_signature(message, satoshi_public_key, user_id)

# ����7������sֵΪ����
if s < 0:
s += N

return (r, s)


# ��ʾ����
if __name__ == "__main__":
print("=== SM2����ǩ��α����ʾ ===")

# ����ʾ���б��ϵ���Կ�ԣ�ʵ���н���Կ������
satoshi_private_key = random.randint(1, N - 1)
satoshi_public_key = elliptic_multiply(satoshi_private_key, G)
print(f"�б��Ϲ�Կ: (x={hex(satoshi_public_key[0])[:16]}..., y={hex(satoshi_public_key[1])[:16]}...)")

# Ҫǩ������Ϣ
message = "I am Satoshi Nakamoto. This message is authentic."
print(f"\n��Ϣ: {message}")

# ������ʵǩ������Ϊ�Աȣ�
satoshi_sm2 = SM2(private_key = satoshi_private_key, public_key = satoshi_public_key,
    user_id = "SATOSHI@BITCOIN.COM")
    real_signature = satoshi_sm2.sign(message)
    r_real, s_real = real_signature
    print(f"\n��ʵǩ��: r={hex(r_real)[:16]}..., s={hex(s_real)[:16]}...")
    print(f"��ʵǩ����֤: {'��Ч' if satoshi_sm2.verify(message, real_signature) else '��Ч'}")

    # α��ǩ����ʹ��������ķ�����
    print("\n=== ʹ��������ķ���α��ǩ�� ===")
    forged_signature = forge_satoshi_signature(message, satoshi_public_key, "SATOSHI@BITCOIN.COM")

    if forged_signature:
r_forge, s_forge = forged_signature
print(f"α��ǩ��: r={hex(r_forge)[:16]}..., s={hex(s_forge)[:16]}...")

# ��֤α���ǩ��
verifier = SM2(public_key = satoshi_public_key, user_id = "SATOSHI@BITCOIN.COM")
is_valid = verifier.verify(message, forged_signature)
print(f"α��ǩ����֤: {'��Ч' if is_valid else '��Ч'}")

if is_valid:
print("�ɹ���SM2ǩ��α����ʾ���")
else:
print("α��ʧ�ܣ�ǩ����Ч")
    else:
print("�޷�����α��ǩ��")*/