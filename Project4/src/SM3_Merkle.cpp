/*#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
using namespace std;

// 二进制转换为十六进制
string BinToHex(string str) {
    string hex = "";
    int temp = 0;
    while (str.size() % 4 != 0) {
        str = "0" + str;
    }
    for (size_t i = 0; i < str.size(); i += 4) {
        if (i + 3 >= str.size()) break;

        temp = (str[i] - '0') * 8 + (str[i + 1] - '0') * 4 +
            (str[i + 2] - '0') * 2 + (str[i + 3] - '0') * 1;
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
    string table[16] = { "0000","0001","0010","0011","0100","0101","0110","0111",
                         "1000","1001","1010","1011","1100","1101","1110","1111" };
    for (char c : str) {
        if (c >= 'A' && c <= 'F') {
            bin += table[c - 'A' + 10];
        }
        else if (c >= 'a' && c <= 'f') {  // 新增：支持小写十六进制
            bin += table[c - 'a' + 10];
        }
        else if (c >= '0' && c <= '9') {
            bin += table[c - '0'];
        }
    }
    return bin;
}

// 二进制转换为十进制
long long BinToDec(string str) {
    long long dec = 0;
    for (size_t i = 0; i < str.size(); i++) {
        if (str[i] != '0' && str[i] != '1') return 0;
        dec += (str[i] - '0') * pow(2, str.size() - i - 1);
    }
    return dec;
}

// 十进制转换为二进制
string DecToBin(long long num) {
    if (num == 0) return "0";
    string bin = "";
    while (num > 0) {
        bin = to_string(num % 2) + bin;
        num = num / 2;
    }
    return bin;
}

// 十六进制转换为十进制
long long HexToDec(string str) {
    long long dec = 0;
    for (size_t i = 0; i < str.size(); i++) {
        long long val;
        if (str[i] >= 'A' && str[i] <= 'F') {
            val = str[i] - 'A' + 10;
        }
        else if (str[i] >= 'a' && str[i] <= 'f') {  // 新增：支持小写十六进制
            val = str[i] - 'a' + 10;
        }
        else if (str[i] >= '0' && str[i] <= '9') {
            val = str[i] - '0';
        }
        else {
            return 0;
        }
        dec += val * pow(16, str.size() - i - 1);
    }
    return dec;
}

// 十进制转换为十六进制
string DecToHex(long long num) {
    if (num == 0) return "0";
    string hex = "";
    int temp = 0;
    while (num > 0) {
        temp = num % 16;
        if (temp < 10 && temp >= 0) {
            hex = to_string(temp) + hex;
        }
        else {
            hex += ('A' + (temp - 10));
        }
        num = num / 16;
    }
    return hex;
}

// 循环左移
string LeftShift(string str, int len) {
    if (len <= 0 || str.empty()) return str;
    string res = HexToBin(str);
    len %= res.size();
    res = res.substr(len) + res.substr(0, len);
    return BinToHex(res);
}

// 异或操作
string XOR(string str1, string str2) {
    string res1 = HexToBin(str1);
    string res2 = HexToBin(str2);

    if (res1.size() > res2.size()) {
        res2 = string(res1.size() - res2.size(), '0') + res2;
    }
    else if (res2.size() > res1.size()) {
        res1 = string(res2.size() - res1.size(), '0') + res1;
    }

    string res = "";
    for (size_t i = 0; i < res1.size(); i++) {
        res += (res1[i] == res2[i]) ? "0" : "1";
    }
    return BinToHex(res);
}

// 与操作
string AND(string str1, string str2) {
    string res1 = HexToBin(str1);
    string res2 = HexToBin(str2);

    if (res1.size() > res2.size()) {
        res2 = string(res1.size() - res2.size(), '0') + res2;
    }
    else if (res2.size() > res1.size()) {
        res1 = string(res2.size() - res1.size(), '0') + res1;
    }

    string res = "";
    for (size_t i = 0; i < res1.size(); i++) {
        res += ((res1[i] == '1') && (res2[i] == '1')) ? "1" : "0";
    }
    return BinToHex(res);
}

// 或操作
string OR(string str1, string str2) {
    string res1 = HexToBin(str1);
    string res2 = HexToBin(str2);

    if (res1.size() > res2.size()) {
        res2 = string(res1.size() - res2.size(), '0') + res2;
    }
    else if (res2.size() > res1.size()) {
        res1 = string(res2.size() - res1.size(), '0') + res1;
    }

    string res = "";
    for (size_t i = 0; i < res1.size(); i++) {
        res += ((res1[i] == '0') && (res2[i] == '0')) ? "0" : "1";
    }
    return BinToHex(res);
}

// 非操作
string NOT(string str) {
    string res1 = HexToBin(str);
    string res = "";
    for (char c : res1) {
        res += (c == '0') ? "1" : "0";
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

    while (res1.size() < 32) res1 = "0" + res1;
    while (res2.size() < 32) res2 = "0" + res2;
    if (res1.size() > 32) res1 = res1.substr(res1.size() - 32);
    if (res2.size() > 32) res2 = res2.substr(res2.size() - 32);

    char temp = '0';
    string res = "";
    for (int i = 31; i >= 0; i--) {
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

// 消息扩展 - 修复核心问题：确保正确生成扩展数据
string extension(string str) {
    // 确保输入是128字符（64字节）
    if (str.size() != 128) {
        cerr << "Error: Invalid message length for extension: " << str.size() << endl;
        return "";
    }

    string res = str;
    // 初始化W[0..15]
    vector<string> W(68);
    for (int i = 0; i < 16; i++) {
        W[i] = str.substr(i * 8, 8);
    }

    // 计算W[16..67]
    for (int i = 16; i < 68; i++) {
        string s1 = XOR(W[i - 16], W[i - 9]);
        s1 = XOR(P1(s1), LeftShift(W[i - 3], 15));
        string s2 = XOR(LeftShift(W[i - 13], 7), W[i - 6]);
        W[i] = XOR(s1, s2);
        res += W[i];
    }

    // 计算W'[0..63]
    for (int i = 0; i < 64; i++) {
        res += XOR(W[i], W[i + 4]);
    }

    return res;
}

// 消息压缩 - 修复越界访问问题
string compress(string extensionB, string V) {
    // 检查初始向量长度
    if (V.size() != 64) {
        cerr << "Error: Invalid IV length: " << V.size() << endl;
        return "";
    }

    // 检查扩展数据长度是否正确 (68*8 + 64*8 = 1056)
    if (extensionB.size() != 1056) {
        cerr << "Error: Invalid extension length: " << extensionB.size() << endl;
        return "";
    }

    string A = V.substr(0, 8), B = V.substr(8, 8), C = V.substr(16, 8), D = V.substr(24, 8);
    string E = V.substr(32, 8), F = V.substr(40, 8), G = V.substr(48, 8), H = V.substr(56, 8);
    string SS1, SS2, TT1, TT2;

    for (int j = 0; j < 64; j++) {
        // 从扩展数据中提取Wj和W'j
        string Wj = extensionB.substr(j * 8, 8);
        string Wj_prime = extensionB.substr((j + 68) * 8, 8);

        SS1 = LeftShift(ModAdd(ModAdd(LeftShift(A, 12), E), LeftShift(T(j), (j % 32))), 7);
        SS2 = XOR(SS1, LeftShift(A, 12));
        TT1 = ModAdd(ModAdd(ModAdd(FF(A, B, C, j), D), SS2), Wj_prime);
        TT2 = ModAdd(ModAdd(ModAdd(GG(E, F, G, j), H), SS1), Wj);

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

    // 将结果与初始向量异或
    return XOR(A, V.substr(0, 8)) + XOR(B, V.substr(8, 8)) +
        XOR(C, V.substr(16, 8)) + XOR(D, V.substr(24, 8)) +
        XOR(E, V.substr(32, 8)) + XOR(F, V.substr(40, 8)) +
        XOR(G, V.substr(48, 8)) + XOR(H, V.substr(56, 8));
}

// 数据填充
string padding(string str) {
    string res = "";
    for (char c : str) {
        res += DecToHex(static_cast<unsigned char>(c));
    }

    long long res_length = res.size() * 4;
    res += "8";

    while (res.size() % 128 != 112) {
        res += "0";
    }

    string res_len = DecToHex(res_length);
    while (res_len.size() < 16) {
        res_len = "0" + res_len;
    }
    if (res_len.size() > 16) {
        res_len = res_len.substr(res_len.size() - 16);
    }
    res += res_len;

    return res;
}

// 迭代压缩
string iteration(string str) {
    if (str.empty()) return "";

    int num = str.size() / 128;
    string V = "7380166F4914B2B9172442D7DA8A0600A96F30BC163138AAE38DEE4DB0FB0E4E";

    for (int i = 0; i < num; i++) {
        string B = str.substr(i * 128, 128);
        string extensionB = extension(B);
        if (extensionB.empty()) return "";

        string compressB = compress(extensionB, V);
        if (compressB.empty()) return "";

        V = compressB;
    }
    return V;
}

// 转换为小写
string toLower(string str) {
    for (char& c : str) {
        if (c >= 'A' && c <= 'F') {
            c += 32;
        }
    }
    return str;
}

// 针对十六进制字符串的填充
string padding_hex(string hex_str) {
    for (char c : hex_str) {
        if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f'))) {
            cerr << "Error: Invalid hex character: " << c << endl;
            return "";
        }
    }

    long long n = hex_str.size() * 4;
    string res = hex_str + "80";

    long long temp = (n + 8) % 512;
    long long k = (448 - temp + 512) % 512;
    res += string(k / 4, '0');

    string len_hex = "";
    long long temp_n = n;
    for (int i = 0; i < 16; i++) {
        len_hex = DecToHex(temp_n & 0xF) + len_hex;
        temp_n >>= 4;
    }
    res += len_hex;

    return res;
}

// 针对十六进制字符串的迭代压缩
string iteration_hex(string hex_str) {
    if (hex_str.empty()) return "";

    int num = hex_str.size() / 128;
    string V = "7380166F4914B2B9172442D7DA8A0600A96F30BC163138AAE38DEE4DB0FB0E4E";

    for (int i = 0; i < num; i++) {
        string B = hex_str.substr(i * 128, 128);
        string extensionB = extension(B);
        if (extensionB.empty()) return "";

        string compressB = compress(extensionB, V);
        if (compressB.empty()) return "";

        V = compressB;
    }
    return V;
}

// Merkle树节点哈希
string merkle_hash(string left, string right) {
    string concat = left + right;
    string pad_concat = padding_hex(concat);
    if (pad_concat.empty()) return "";

    string hash = iteration_hex(pad_concat);
    return hash;
}

// 显示进度条
void display_progress(int current, int total, const string& stage) {
    if (total <= 0) return;

    const int bar_length = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(progress * bar_length);

    cout << "\r" << stage << " ";
    cout << setw(3) << static_cast<int>(progress * 100) << "% [";
    for (int i = 0; i < bar_length; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << current << "/" << total;
    cout.flush();
}

// 构建Merkle树
vector<vector<string>> build_merkle_tree(int leaf_count) {
    vector<vector<string>> layers;
    if (leaf_count <= 0) {
        cerr << "Error: Invalid leaf count" << endl;
        return layers;
    }

    vector<string> leaves;
    leaves.reserve(leaf_count);

    cout << "Generating leaf nodes..." << endl;
    for (int i = 0; i < leaf_count; i++) {
        string data = to_string(i);
        string padded = padding(data);
        if (padded.empty()) {
            cerr << "Error: Padding failed for leaf " << i << endl;
            return layers;
        }

        string hash = iteration(padded);
        if (hash.empty()) {
            cerr << "Error: Hashing failed for leaf " << i << endl;
            return layers;
        }

        leaves.push_back(toLower(hash));

        if (i % 100 == 0 || i == leaf_count - 1) {
            display_progress(i + 1, leaf_count, "Leaves:");
        }
    }
    cout << endl;
    layers.push_back(leaves);

    vector<string> current = leaves;
    int layer_index = 1;

    while (current.size() > 1) {
        vector<string> next_level;
        next_level.reserve((current.size() + 1) / 2);

        int total_nodes = (current.size() + 1) / 2;

        for (size_t i = 0; i < current.size(); i += 2) {
            string parent;
            if (i + 1 < current.size()) {
                parent = toLower(merkle_hash(current[i], current[i + 1]));
                if (parent.empty()) {
                    cerr << "Error: Hashing failed for parent node at index " << i / 2 << endl;
                    return layers;
                }
            }
            else {
                parent = current[i];
            }

            next_level.push_back(parent);

            int current_node = (i / 2) + 1;
            display_progress(current_node, total_nodes, "Layer " + to_string(layer_index) + ":");
        }

        cout << endl;
        layers.push_back(next_level);
        current = next_level;
        layer_index++;
    }

    return layers;
}

// 存在性证明
vector<string> generate_existence_proof(const vector<vector<string>>& layers, int index) {
    vector<string> proof;
    if (layers.empty() || index < 0 || static_cast<size_t>(index) >= layers[0].size()) {
        return proof;
    }

    int current_index = index;
    for (size_t i = 0; i < layers.size() - 1; i++) {
        if (static_cast<size_t>(current_index) >= layers[i].size()) {
            break;
        }

        if (current_index % 2 == 1) {
            if (current_index - 1 >= 0) {
                proof.push_back(layers[i][current_index - 1]);
            }
        }
        else if (current_index + 1 < static_cast<int>(layers[i].size())) {
            proof.push_back(layers[i][current_index + 1]);
        }

        current_index /= 2;
    }
    return proof;
}

// 验证存在性证明
bool verify_existence_proof(string leaf_hash, const vector<string>& proof, string root, int index, int leaf_count) {
    if (leaf_hash.empty() || root.empty() || index < 0 || index >= leaf_count) {
        return false;
    }

    string current_hash = leaf_hash;
    int current_index = index;

    for (const string& sibling : proof) {
        if (current_index % 2 == 1) {
            current_hash = toLower(merkle_hash(sibling, current_hash));
        }
        else {
            current_hash = toLower(merkle_hash(current_hash, sibling));
        }

        if (current_hash.empty()) {
            return false;
        }

        current_index /= 2;
    }

    return current_hash == root;
}

// 生成不存在性证明
vector<string> generate_absence_proof(const vector<vector<string>>& layers, const string& target, int leaf_count) {
    vector<string> proof;
    if (layers.empty() || leaf_count <= 0) {
        return proof;
    }

    int target_value;
    try {
        size_t pos;
        target_value = stoi(target, &pos);
        if (pos != target.size()) {
            throw invalid_argument("Invalid target format");
        }
    }
    catch (const exception& e) {
        cerr << "Error: Invalid target value - " << e.what() << endl;
        return proof;
    }

    if (target_value < 0 || target_value >= leaf_count) {
        if (leaf_count > 0) {
            proof.push_back(layers[0][0]);
            proof.push_back(layers[0][leaf_count - 1]);
        }
        return proof;
    }

    return proof;
}

// 主函数
int main() {
    try {
        const int LEAF_COUNT = 10000;

        cout << "Building Merkle Tree with " << LEAF_COUNT << " leaves..." << endl;
        vector<vector<string>> layers = build_merkle_tree(LEAF_COUNT);

        if (layers.empty()) {
            cerr << "Error: Failed to build Merkle Tree" << endl;
            return 1;
        }

        string root = layers.back()[0];
        cout << "\nMerkle Root: " << root << endl << endl;

        int test_index = 123;
        if (test_index < 0 || static_cast<size_t>(test_index) >= layers[0].size()) {
            cerr << "Error: Test index out of range" << endl;
            return 1;
        }

        vector<string> existence_proof = generate_existence_proof(layers, test_index);
        string leaf_hash = layers[0][test_index];

        bool is_valid = verify_existence_proof(leaf_hash, existence_proof, root, test_index, LEAF_COUNT);
        cout << "Existence Proof for leaf " << test_index << ":\n";
        cout << (is_valid ? "VALID" : "INVALID") << endl << endl;

        string test_absence_target = "10000";
        vector<string> absence_proof = generate_absence_proof(layers, test_absence_target, LEAF_COUNT);
        cout << "Absence Proof for target " << test_absence_target << ":\n";
        if (!absence_proof.empty()) {
            cout << "Target is outside the range of leaf values [0, " << LEAF_COUNT - 1 << "]\n";
        }
        else {
            cout << "Target may exist in the tree\n";
        }
    }
    catch (const exception& e) {
        cerr << "Runtime error: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        cerr << "Unknown error occurred" << endl;
        return 1;
    }

    return 0;
}
*/