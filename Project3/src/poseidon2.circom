pragma circom 2.1.6;

template Sbox() {
    signal input in;
    signal output out;
    
    signal in2;
    signal in4;
    signal out5;
    
    in2 <== in * in;
    in4 <== in2 * in2;
    out5 <== in4 * in;
    out <== out5;
}

template MixLayer() {
    signal input in[2];
    signal output out[2];
    
    // MDS 矩阵 (示例值)
    var M00 = 3;
    var M01 = 1;
    var M10 = 1;
    var M11 = 2;
    
    out[0] <== M00 * in[0] + M01 * in[1];
    out[1] <== M10 * in[0] + M11 * in[1];
}

template Round(roundConstants) {
    signal input in[2];
    signal output out[2];
    
    // 添加轮常数
    signal afterAddRC[2];
    afterAddRC[0] <== in[0] + roundConstants[0];
    afterAddRC[1] <== in[1] + roundConstants[1];
    
    // S-box 应用
    component sbox0 = Sbox();
    sbox0.in <== afterAddRC[0];
    
    component sbox1 = Sbox();
    sbox1.in <== afterAddRC[1];
    
    // 混合层
    component mix = MixLayer();
    mix.in[0] <== sbox0.out;
    mix.in[1] <== sbox1.out;
    
    out[0] <== mix.out[0];
    out[1] <== mix.out[1];
}

template Poseidon2(roundConstants) {
    signal input in;
    signal output out;
    
    // 初始化状态: [input, capacity=0]
    signal state[2];
    state[0] <== in;
    state[1] <== 0;
    
    // 应用轮函数
    for (var i = 0; i < 5; i++) {
        component round = Round(roundConstants[i]);
        round.in[0] <== state[0];
        round.in[1] <== state[1];
        
        state[0] <== round.out[0];
        state[1] <== round.out[1];
    }
    
    // 输出哈希值
    out <== state[0];
}

template Groth16Poseidon2() {
    // 隐私输入: 原始消息
    signal private input message;
    
    // 公开输出: Poseidon2 哈希值
    signal output hash;
    
    // 加载轮常数参数
    include "circuits/poseidon2_params.circom";
    
    // 计算 Poseidon2 哈希
    component poseidon = Poseidon2(ROUND_CONSTANTS);
    poseidon.in <== message;
    
    hash <== poseidon.out;
}

component main {public [hash]} = Groth16Poseidon2();
