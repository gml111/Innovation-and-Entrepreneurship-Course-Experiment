import random
import hashlib
import math


# 模拟的同态加密系统（仅供演示）
class SimpleHomomorphicEncryption:
    def __init__(self):
        # 生成大质数 p 和 q
        self.p = self.generate_large_prime(1024)
        self.q = self.generate_large_prime(1024)
        self.n = self.p * self.q
        self.nsquare = self.n * self.n

    def generate_large_prime(self, bits):
        """生成大质数（简化实现）"""
        while True:
            p = random.randint(2 ** (bits - 1), 2 ** bits)
            if p % 2 != 0 and self.is_prime(p):
                return p

    def is_prime(self, n, k=10):
        """米勒-拉宾素数测试"""
        if n <= 1:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1

        for _ in range(k):
            a = random.randint(2, n - 2)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def encrypt(self, plaintext):
        """模拟加密"""
        r = random.randint(1, self.n - 1)
        # Paillier 加密公式的简化模拟: (1 + plaintext * n) * r^n mod n^2
        ciphertext = (1 + plaintext * self.n) * pow(r, self.n, self.nsquare) % self.nsquare
        return {'ciphertext': ciphertext, 'r': r}

    def decrypt(self, ciphertext):
        """模拟解密"""
        # 简化解密过程
        return (ciphertext - 1) // self.n

    def raw_decrypt(self, ciphertext):
        """直接解密整数密文"""
        return (ciphertext - 1) // self.n


# 使用简化的哈希函数
def hash_to_point(u: str) -> bytes:
    """将字符串映射为固定长度的哈希值"""
    return hashlib.sha256(u.encode()).digest()


# 简化的点运算
def point_operation(value: bytes, key: int) -> bytes:
    """使用密钥对哈希值进行处理"""
    key_bytes = key.to_bytes(32, 'big')
    # 使用异或操作和哈希组合
    transformed = bytes([v ^ k for v, k in zip(value, key_bytes)])
    return hashlib.sha256(transformed).digest()


class Party1:
    def __init__(self, set_V: list):
        self.set_V = set_V
        self.k1 = random.randint(1, 2 ** 256 - 1)  # 256位随机密钥
        self.paillier_pubkey = None
        self.intersection_sum_cipher = None

    def set_paillier_pubkey(self, pubkey):
        self.paillier_pubkey = pubkey

    def round1(self) -> list:
        # 计算 H(v_i)^k1
        self.hashed_exponents = []
        for v in self.set_V:
            P = hash_to_point(v)
            Q = point_operation(P, self.k1)
            self.hashed_exponents.append(Q)

        # 随机洗牌
        random.shuffle(self.hashed_exponents)
        return self.hashed_exponents

    def round3(self, Z: list, received_set: list):
        # 将Z集合转换为字节串集合
        Z_bytes_set = set(Z)

        # 处理接收的元组 [(H(w_j)^k2, enc(t_j)]
        intersection_ciphers = []
        for point_bytes, enc_t in received_set:
            # 计算 H(w_j)^{k1*k2}
            T = point_operation(point_bytes, self.k1)

            # 检查是否在Z中
            if T in Z_bytes_set:
                intersection_ciphers.append(enc_t)

        # 同态求和
        if intersection_ciphers:
            product = 1
            nsquare = self.paillier_pubkey.nsquare
            # 将所有密文相乘
            for c in intersection_ciphers:
                product = (product * c) % nsquare

            # 刷新密文 (重新随机化)
            r = random.randint(1, self.paillier_pubkey.n - 1)
            c0 = pow(r, self.paillier_pubkey.n, nsquare)  # 加密0
            refreshed_cipher = (product * c0) % nsquare
        else:
            # 如果交集为空，加密0
            refreshed_cipher = self.paillier_pubkey.encrypt(0)['ciphertext']

        self.intersection_sum_cipher = refreshed_cipher
        return refreshed_cipher


class Party2:
    def __init__(self, set_W: list):
        self.set_W = set_W
        self.k2 = random.randint(1, 2 ** 256 - 1)  # 256位随机密钥
        # 生成同态加密密钥
        self.paillier_public_key = SimpleHomomorphicEncryption()
        self.paillier_private_key = self.paillier_public_key
        self.intersection_sum = None

    def get_paillier_pubkey(self):
        return self.paillier_public_key

    def round2(self, received_points: list) -> tuple:
        # 步骤1: 计算 H(v_i)^{k1*k2}
        Z = []
        for P in received_points:
            Q = point_operation(P, self.k2)
            Z.append(Q)
        random.shuffle(Z)

        # 步骤2: 处理自己的集合
        processed_set = []
        for w, t in self.set_W:
            # 计算 H(w_j)^{k2}
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

    # 第1轮: P1 -> P2
    r1_output = p1.round1()

    # 第2轮: P2 -> P1 (发送Z和处理后的集合)
    Z, r2_output = p2.round2(r1_output)

    # 第3轮: P1 -> P2 (发送加密的和)
    sum_cipher = p1.round3(Z, r2_output)

    # P2解密得到结果
    result = p2.decrypt_sum(sum_cipher)

    print(f"交集元素: id1, id2, id3 (预期和为60)")
    print(f"实际计算得到的交集和: {result}")