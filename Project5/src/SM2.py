import os
import random
import binascii
from gmssl import sm3, func

# ===== 调试模式配置 =====
DEBUG_FIXED_KEYS = True  # 固定私钥/公钥
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


def mod_inv(a, modulus=PRIME):
    """扩展欧几里得算法求模逆"""
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
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return elliptic_double(point1)
    if point1 == (0, 0):
        return point2
    if point2 == (0, 0):
        return point1
    if point1[0] == point2[0]:
        return (0, 0)

    # 计算斜率
    dx = (point2[0] - point1[0]) % PRIME
    dy = (point2[1] - point1[1]) % PRIME
    s = dy * mod_inv(dx, PRIME)
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
    s = numerator * mod_inv(denominator, PRIME)

    x = (s ** 2 - 2 * point[0]) % PRIME
    y = (s * (point[0] - x) - point[1]) % PRIME
    return (x, y)


def elliptic_multiply(k, point):
    """椭圆曲线点乘法（标量乘法）"""
    if k % N == 0 or point == (0, 0):
        return (0, 0)
    if k < 0:
        return elliptic_multiply(-k, point)

    # 二进制展开
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


class SM2:
    def __init__(self, private_key=None, public_key=None, user_id=None):
        self.user_id = user_id

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
        self.za = get_za(self.user_id, self.public_key) if self.public_key else None

    def generate_key_pair(self):
        """生成密钥对"""
        self.private_key = random.randint(1, N - 1)
        self.public_key = elliptic_multiply(self.private_key, G)
        self.za = get_za(self.user_id, self.public_key)
        return self.private_key, self.public_key

    def sign(self, message):
        """SM2签名"""
        if not self.private_key:
            raise ValueError("Private key is required for signing")

        e = self._hash_message(message)
        if DEBUG_FIXED_K:
            k = FIXED_K % N
        else:
            k = random.randint(1, N - 1)

        # 计算椭圆曲线点[k]G
        x1, y1 = elliptic_multiply(k, G)
        r = (e + x1) % N

        # 避免r为0或r+k==n的情况
        if r == 0 or r + k == N:
            return self.sign(message)

        # 计算s = (1 + d)^{-1} * (k - r * d) mod n
        d = self.private_key
        s = (pow(1 + d, N - 2, N) * (k - r * d)) % N

        # 避免s为0
        if s == 0:
            return self.sign(message)

        return (r, s)

    def verify(self, message, signature):
        """SM2验证签名"""
        if not self.public_key:
            raise ValueError("Public key is required for verification")

        r, s = signature

        # 检查r和s是否在[1, n-1]范围内
        if not (1 <= r <= N - 1 and 1 <= s <= N - 1):
            return False

        e = self._hash_message(message)
        t = (r + s) % N

        # 计算椭圆曲线点[s]G + [t]P
        point1 = elliptic_multiply(s, G)
        point2 = elliptic_multiply(t, self.public_key)
        x1, y1 = elliptic_add(point1, point2)

        # 验证R = (e + x1) mod n
        return (r % N) == ((e + x1) % N)

    def _hash_message(self, message):
        """计算消息哈希e = Hash(Z_A || M)"""
        if not self.za:
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
    sm2 = SM2(user_id="test@example.com")
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
    sm2_custom = SM2(user_id="Alice@email.com")
    signature_custom = sm2_custom.sign(message)
    verify_result_custom = sm2_custom.verify(message, signature_custom)
    print(f"自定义ID签名验证: {'成功' if verify_result_custom else '失败'}")

    # 验证不同用户ID的签名是否相同
    same_signature = signature == signature_custom
    print(f"不同用户ID的签名是否相同: {'是' if same_signature else '否'}")

    print("\n所有测试完成!")