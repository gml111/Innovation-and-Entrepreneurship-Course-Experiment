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


def mod_inv(a, modulus=N):
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
    if point1 == (0, 0):
        return point2
    if point2 == (0, 0):
        return point1
    if point1 == point2:
        return elliptic_double(point1)
    if point1[0] == point2[0]:
        return (0, 0)  # 无穷远点

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
    if user_id is None:
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
        self.private_key = private_key
        self.public_key = public_key

        if self.private_key and not self.public_key:
            self.public_key = elliptic_multiply(self.private_key, G)

        if self.public_key:
            self.za = get_za(self.user_id, self.public_key)
        else:
            self.za = None

    def sign(self, message, k=None):
        """生成SM2签名"""
        if not self.private_key:
            raise ValueError("需要私钥才能签名")

        e = compute_hash(self.za, message)

        if k is None:
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

        if s == 0:
            return self.sign(message)

        return (r, s)

    def verify(self, message, signature):
        """验证SM2签名"""
        if not self.public_key:
            raise ValueError("需要公钥才能验证签名")

        r, s = signature

        if not (1 <= r <= N - 1 and 1 <= s <= N - 1):
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


def forge_satoshi_signature(message, satoshi_public_key, user_id="SATOSHI@BITCOIN.COM"):
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

    # 步骤6：构造s = [ (r / v) - u ] mod N
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
    satoshi_sm2 = SM2(private_key=satoshi_private_key, public_key=satoshi_public_key,
                      user_id="SATOSHI@BITCOIN.COM")
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
        verifier = SM2(public_key=satoshi_public_key, user_id="SATOSHI@BITCOIN.COM")
        is_valid = verifier.verify(message, forged_signature)
        print(f"伪造签名验证: {'有效' if is_valid else '无效'}")

        if is_valid:
            print("成功！SM2签名伪造演示完成")
        else:
            print("伪造失败，签名无效")
    else:
        print("无法生成伪造签名")