import random
from math import gcd, ceil, log
from gmssl import sm3, func
import binascii

# ===== 调试模式配置 =====
DEBUG_FIXED_KEYS = True  # 固定私钥/公钥
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


def mod_inv(a, modulus=N):
    """扩展欧几里得算法求模逆，使用N作为默认模数"""
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % modulus, modulus
    while low > 1:
        r = high // low
        nm, new = hm - lm * r, high - low * r
        lm, low, hm, high = nm, new, lm, low
    return lm % modulus


def elliptic_add(point1, point2):
    """椭圆曲线点加法"""
    if point1 == (0, 0):
        return point2
    if point2 == (0, 0):
        return point1
    if point1 == point2:
        return elliptic_double(point1)
    if point1[0] == point2[0]:
        return (0, 0)  # 无穷远点

    # 计算斜率
    dx = (point2[0] - point1[0]) % PRIME
    dy = (point2[1] - point1[1]) % PRIME
    s = dy * mod_inv(dx, PRIME) % PRIME
    x = (s ** 2 - point1[0] - point2[0]) % PRIME
    y = (s * (point1[0] - x) - point1[1]) % PRIME
    return (x, y)


def elliptic_double(point):
    """椭圆曲线倍点运算"""
    if point == (0, 0):
        return point

    # 计算斜率
    numerator = (3 * point[0] ** 2 + A) % PRIME
    denominator = (2 * point[1]) % PRIME
    s = numerator * mod_inv(denominator, PRIME) % PRIME

    x = (s ** 2 - 2 * point[0]) % PRIME
    y = (s * (point[0] - x) - point[1]) % PRIME
    return (x, y)


def elliptic_multiply(k, point):
    """椭圆曲线点乘法（标量乘法）"""
    if k % N == 0 or point == (0, 0):
        return (0, 0)
    if k < 0:
        return elliptic_multiply(-k, point)

    # 二进制展开法
    result = (0, 0)
    addend = point

    while k:
        if k & 1:
            result = elliptic_add(result, addend)
        addend = elliptic_double(addend)
        k >>= 1

    return result


def get_za(user_id, public_key):
    """计算Z_A值（SM2预计算哈希部分）"""
    # 用户ID默认值
    if user_id is None:
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


def compute_hash(za, message):
    """计算消息哈希e = Hash(Z_A || M)"""
    msg_bytes = message.encode('utf-8')
    za_bytes = bytes.fromhex(za)
    input_data = func.bytes_to_list(za_bytes + msg_bytes)
    hash_hex = sm3.sm3_hash(input_data)
    return int(hash_hex, 16) % N


class SM2:
    def __init__(self, private_key=None, public_key=None, user_id=None):
        self.user_id = user_id or "ALICE123@YAHOO.COM"
        self.curve_params = {
            'p': PRIME,
            'a': A,
            'b': B,
            'n': N,
            'Gx': Gx,
            'Gy': Gy
        }

        if DEBUG_FIXED_KEYS:
            self.private_key = FIXED_PRIVATE_KEY
            self.public_key = elliptic_multiply(self.private_key, G)
        else:
            self.private_key = private_key
            if private_key:
                self.public_key = elliptic_multiply(private_key, G)
            else:
                self.public_key = public_key if public_key else None

        # 计算并存储ZA
        if self.public_key:
            self.za = get_za(self.user_id, self.public_key)
        else:
            self.za = None

    def generate_key_pair(self):
        """生成密钥对"""
        self.private_key = random.randint(1, N - 1)
        self.public_key = elliptic_multiply(self.private_key, G)
        self.za = get_za(self.user_id, self.public_key)
        return self.private_key, self.public_key

    def sign(self, message, k=None):
        """SM2签名"""
        if not self.private_key:
            raise ValueError("Private key is required for signing")

        e = compute_hash(self.za, message)

        if k is None:
            if DEBUG_FIXED_K:
                k = FIXED_K % N
            else:
                k = random.randint(1, N - 1)
        else:
            k = k % N

        # 计算椭圆曲线点[k]G
        x1, y1 = elliptic_multiply(k, G)
        r = (e + x1) % N

        # 避免r为0或r+k==n的情况
        if r == 0 or (r + k) % N == 0:
            return self.sign(message, k)

        # 计算s = (1 + d)^{-1} * (k - r * d) mod n
        d = self.private_key
        s = mod_inv(1 + d, N) * (k - r * d) % N

        # 确保s为正数
        if s < 0:
            s += N

        # 避免s为0
        if s == 0:
            return self.sign(message, k)

        return (r, s)

    def verify(self, message, signature, public_key=None, user_id=None):
        """SM2验证签名"""
        if public_key is None:
            public_key = self.public_key
        if public_key is None:
            raise ValueError("Public key is required for verification")

        user_id = user_id or self.user_id
        za = get_za(user_id, public_key)

        r, s = signature

        # 检查r和s是否在[1, n-1]范围内
        if not (1 <= r <= N - 1 and 1 <= s <= N - 1):
            return False

        e = compute_hash(za, message)
        t = (r + s) % N
        if t == 0:
            return False

        # 计算椭圆曲线点[s]G + [t]P
        sG = elliptic_multiply(s, G)
        tP = elliptic_multiply(t, public_key)
        x1, y1 = elliptic_add(sG, tP)

        # 验证R = (e + x1) mod n
        R = (e + x1) % N
        return R == r

    # ===== 攻击演示方法 =====
    def k_leakage_attack(self, signature, k, message, public_key=None, user_id=None):
        """已知k值推导私钥的攻击演示"""
        if public_key is None:
            public_key = self.public_key
        if public_key is None:
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
        if inv_denom is None:
            return None

        private_key_guess = (numerator * inv_denom) % N
        # 确保结果为正数
        if private_key_guess < 0:
            private_key_guess += N
        return private_key_guess

    def repeated_k_attack(self, signature1, message1, signature2, message2, public_key=None, user_id=None):
        """相同k值对多消息签名的攻击演示 - 修复k值推导公式"""
        if public_key is None:
            public_key = self.public_key
        if public_key is None:
            raise ValueError("Public key required for attack")

        user_id = user_id or self.user_id
        za = get_za(user_id, public_key)

        r1, s1 = signature1
        r2, s2 = signature2

        # 计算两条消息的哈希值
        e1 = compute_hash(za, message1)
        e2 = compute_hash(za, message2)

        # 重新推导k值计算公式：k = (e1 - e2 + s1*r1 - s2*r2) * inv(s2 - s1) mod n
        numerator_k = (e1 - e2 + s1 * r1 - s2 * r2) % N
        denominator_k = (s2 - s1) % N

        # 检查分母是否为0
        if denominator_k == 0:
            print("错误：delta_s为0，无法计算模逆")
            return None

        inv_denominator_k = mod_inv(denominator_k)
        if inv_denominator_k is None:
            return None

        k_guess = (numerator_k * inv_denominator_k) % N

        # 推导私钥
        private_key_guess = self.k_leakage_attack(signature1, k_guess, message1, public_key, user_id)
        return private_key_guess

    def multi_user_k_attack(self, signature, message, public_key, user_id, k):
        """多个用户使用相同k值的攻击演示 - 修复私钥推导公式"""
        # 计算Z_A
        za = get_za(user_id, public_key)

        r, s = signature
        e = compute_hash(za, message)

        # 正确的私钥推导公式: d = (k - s*(1 + d)) / r mod n
        # 重新整理为: d = (k - s) * inv(r + s) mod n
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
        if inv_denom is None:
            return None

        private_key_guess = (numerator * inv_denom) % N
        # 确保结果为正数
        if private_key_guess < 0:
            private_key_guess += N
        return private_key_guess


if __name__ == "__main__":
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
    print(f"\n攻击结果 - 推导私钥: ", end="")
    if private_key_guess is not None:
        print(f"{hex(private_key_guess)}")
    else:
        print("无法推导私钥（计算错误）")

    print(f"真实私钥: {hex(private_key)}")
    if private_key_guess is not None and private_key_guess == private_key:
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
    print(f"\n攻击结果 - 推导私钥: ", end="")
    if private_key_guess is not None:
        print(f"{hex(private_key_guess)}")
    else:
        print("无法推导私钥（计算错误）")

    print(f"真实私钥: {hex(private_key)}")
    if private_key_guess is not None and private_key_guess == private_key:
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
    sm2_bob = SM2(private_key=bob_private_key, public_key=bob_public_key, user_id=bob_id)
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

    print(f"\n攻击结果 - Alice私钥推导: ", end="")
    if priv_alice_guess is not None:
        print(f"{hex(priv_alice_guess)}")
    else:
        print("无法推导私钥（计算错误）")

    print(f"真实Alice私钥: {hex(alice_private_key)}")
    if priv_alice_guess is not None and priv_alice_guess == alice_private_key:
        print(">>> Alice私钥攻击成功! <<<")
    else:
        print(">>> Alice私钥攻击失败! <<<")

    print(f"\n攻击结果 - Bob私钥推导: ", end="")
    if priv_bob_guess is not None:
        print(f"{hex(priv_bob_guess)}")
    else:
        print("无法推导私钥（计算错误）")

    print(f"真实Bob私钥: {hex(bob_private_key)}")
    if priv_bob_guess is not None and priv_bob_guess == bob_private_key:
        print(">>> Bob私钥攻击成功! <<<")
    else:
        print(">>> Bob私钥攻击失败! <<<")

    print("\n" + "=" * 80)
    print("攻击演示完成".center(80))
    print("=" * 80)
