/*#include <iostream>
#include <string>
#include <cmath>
using namespace std;

// 二进制转换为十六进制
string BinToHex(string str) {
    string hex = "";
    int temp = 0;
    while (str.size() % 4 != 0) {
        str = "0" + str;
    }
    for (int i = 0; i < str.size(); i += 4) {
        temp = (str[i] - '0') * 8 + (str[i + 1] - '0') * 4 + (str[i + 2] - '0') * 2 + (str[i + 3] - '0') * 1;
        if (temp < 10) {
            hex += to_string(temp);
        }
        else {
            hex += 'A' + (temp - 10);
        }
    }
    return hex;
}

// 十六进制转换为二进制
string HexToBin(string str) {
    string bin = "";
    string table[16] = { "0000","0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111" };
    for (int i = 0; i < str.size(); i++) {
        if (str[i] >= 'A' && str[i] <= 'F') {
            bin += table[str[i] - 'A' + 10];
        }
        else {
            bin += table[str[i] - '0'];
        }
    }
    return bin;
}

// 二进制转换为十进制
int BinToDec(string str) {
    int dec = 0;
    for (int i = 0; i < str.size(); i++) {
        dec += (str[i] - '0') * pow(2, str.size() - i - 1);
    }
    return dec;
}

// 十进制转换为二进制
string DecToBin(int str) {
    string bin = "";
    if (str == 0) return "0";
    while (str >= 1) {
        bin = to_string(str % 2) + bin;
        str = str / 2;
    }
    return bin;
}

// 十六进制转换为十进制
int HexToDec(string str) {
    int dec = 0;
    for (int i = 0; i < str.size(); i++) {
        if (str[i] >= 'A' && str[i] <= 'F') {
            dec += (str[i] - 'A' + 10) * pow(16, str.size() - i - 1);
        }
        else {
            dec += (str[i] - '0') * pow(16, str.size() - i - 1);
        }
    }
    return dec;
}

// 十进制转换为十六进制
string DecToHex(int str) {
    string hex = "";
    int temp = 0;
    if (str == 0) return "0";
    while (str >= 1) {
        temp = str % 16;
        if (temp < 10 && temp >= 0) {
            hex = to_string(temp) + hex;
        }
        else {
            hex += ('A' + (temp - 10));
        }
        str = str / 16;
    }
    return hex;
}

// 数据填充
string padding(string str) {
    string res = "";
    for (int i = 0; i < str.size(); i++) {
        res += DecToHex((int)str[i]);
    }
    int res_length = res.size() * 4;
    res += "8";
    while (res.size() % 128 != 112) {
        res += "0";
    }
    string res_len = DecToHex(res_length);
    while (res_len.size() != 16) {
        res_len = "0" + res_len;
    }
    res += res_len;
    return res;
}

// 循环左移
string LeftShift(string str, int len) {
    string res = HexToBin(str);
    res = res.substr(len) + res.substr(0, len);
    return BinToHex(res);
}

// 异或操作
string XOR(string str1, string str2) {
    string res1 = HexToBin(str1);
    string res2 = HexToBin(str2);
    string res = "";
    for (int i = 0; i < res1.size(); i++) {
        res += (res1[i] == res2[i]) ? "0" : "1";
    }
    return BinToHex(res);
}

// 与操作
string AND(string str1, string str2) {
    string res1 = HexToBin(str1);
    string res2 = HexToBin(str2);
    string res = "";
    for (int i = 0; i < res1.size(); i++) {
        res += ((res1[i] == '1') && (res2[i] == '1')) ? "1" : "0";
    }
    return BinToHex(res);
}

// 或操作
string OR(string str1, string str2) {
    string res1 = HexToBin(str1);
    string res2 = HexToBin(str2);
    string res = "";
    for (int i = 0; i < res1.size(); i++) {
        res += ((res1[i] == '0') && (res2[i] == '0')) ? "0" : "1";
    }
    return BinToHex(res);
}

// 非操作
string NOT(string str) {
    string res1 = HexToBin(str);
    string res = "";
    for (int i = 0; i < res1.size(); i++) {
        res += (res1[i] == '0') ? "1" : "0";
    }
    return BinToHex(res);
}

// 单比特异或
char binXor(char str1, char str2) {
    return (str1 == str2) ? '0' : '1';
}

// 单比特与
char binAnd(char str1, char str2) {
    return ((str1 == '1') && (str2 == '1')) ? '1' : '0';
}

// 模2^32加法
string ModAdd(string str1, string str2) {
    string res1 = HexToBin(str1);
    string res2 = HexToBin(str2);
    char temp = '0';
    string res = "";
    for (int i = res1.size() - 1; i >= 0; i--) {
        res = binXor(binXor(res1[i], res2[i]), temp) + res;
        if (binAnd(res1[i], res2[i]) == '1') {
            temp = '1';
        }
        else {
            temp = (binXor(res1[i], res2[i]) == '1') ? binAnd('1', temp) : '0';
        }
    }
    return BinToHex(res);
}

// 置换P1
string P1(string str) {
    return XOR(XOR(str, LeftShift(str, 15)), LeftShift(str, 23));
}

// 置换P0
string P0(string str) {
    return XOR(XOR(str, LeftShift(str, 9)), LeftShift(str, 17));
}

// Tj常量
string T(int j) {
    return (0 <= j && j <= 15) ? "79CC4519" : "7A879D8A";
}

// 布尔函数FF
string FF(string str1, string str2, string str3, int j) {
    if (0 <= j && j <= 15) {
        return XOR(XOR(str1, str2), str3);
    }
    else {
        return OR(OR(AND(str1, str2), AND(str1, str3)), AND(str2, str3));
    }
}

// 布尔函数GG
string GG(string str1, string str2, string str3, int j) {
    if (0 <= j && j <= 15) {
        return XOR(XOR(str1, str2), str3);
    }
    else {
        return OR(AND(str1, str2), AND(NOT(str1), str3));
    }
}

// 消息扩展
string extension(string str) {
    string res = str;
    for (int i = 16; i < 68; i++) {
        res += XOR(XOR(P1(XOR(XOR(res.substr((i - 16) * 8, 8), res.substr((i - 9) * 8, 8)), LeftShift(res.substr((i - 3) * 8, 8), 15))), LeftShift(res.substr((i - 13) * 8, 8), 7)), res.substr((i - 6) * 8, 8));
    }
    for (int i = 0; i < 64; i++) {
        res += XOR(res.substr(i * 8, 8), res.substr((i + 4) * 8, 8));
    }
    return res;
}

// 消息压缩
string compress(string str1, string str2) {
    string IV = str2;
    string A = IV.substr(0, 8), B = IV.substr(8, 8), C = IV.substr(16, 8), D = IV.substr(24, 8);
    string E = IV.substr(32, 8), F = IV.substr(40, 8), G = IV.substr(48, 8), H = IV.substr(56, 8);
    string SS1, SS2, TT1, TT2;

    for (int j = 0; j < 64; j++) {
        SS1 = LeftShift(ModAdd(ModAdd(LeftShift(A, 12), E), LeftShift(T(j), (j % 32))), 7);
        SS2 = XOR(SS1, LeftShift(A, 12));
        TT1 = ModAdd(ModAdd(ModAdd(FF(A, B, C, j), D), SS2), str1.substr((j + 68) * 8, 8));
        TT2 = ModAdd(ModAdd(ModAdd(GG(E, F, G, j), H), SS1), str1.substr(j * 8, 8));
        D = C;
        C = LeftShift(B, 9);
        B = A;
        A = TT1;
        H = G;
        G = LeftShift(F, 19);
        F = E;
        E = P0(TT2);
    }
    return (A + B + C + D + E + F + G + H);
}

// 迭代压缩
string iteration(string str) {
    int num = str.size() / 128;
    string V = "7380166F4914B2B9172442D7DA8A0600A96F30BC163138AAE38DEE4DB0FB0E4E";
    string B, extensionB, compressB;
    for (int i = 0; i < num; i++) {
        B = str.substr(i * 128, 128);
        extensionB = extension(B);
        compressB = compress(extensionB, V);
        V = XOR(V, compressB);
    }
    return V;
}

// 转换为小写（便于对比）
string toLower(string str) {
    for (int i = 0; i < str.size(); i++) {
        if (str[i] >= 'A' && str[i] <= 'F') {
            str[i] += 32;
        }
    }
    return str;
}

int main() {
    // 测试用例及标准结果（仅保留前三个）
    struct TestCase {
        string input;
        string standardHash;
    };

    TestCase tests[] = {
        {"", "1ab21d8355cfa17f8e61194831e81a8f22bec8c728fefb747ed035eb5082aa2b"},
        {"abc", "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"},
        {"abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd", "debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732"}
    };
    int testCount = sizeof(tests) / sizeof(tests[0]);

    // 执行测试
    for (int i = 0; i < testCount; i++) {
        cout << "测试用例 " << i + 1 << ":" << endl;
        cout << "输入: " << (tests[i].input.empty() ? "<empty>" : tests[i].input) << endl;

        // 计算哈希
        string paddingValue = padding(tests[i].input);
        string result = iteration(paddingValue);
        string resultLower = toLower(result);

        // 输出结果
        cout << "计算哈希: " << resultLower << endl;
        cout << "标准哈希: " << tests[i].standardHash << endl;
        cout << "验证结果: " << (resultLower == tests[i].standardHash ? "PASS" : "FAIL") << endl << endl;
    }

    return 0;
}*/