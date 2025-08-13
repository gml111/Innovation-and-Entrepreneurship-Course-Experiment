# scripts/generate_params.js
const fs = require('fs');
const path = require('path');

function generatePoseidon2Params() {
    // 生成轮常数 (5轮)
    const ROUND_CONSTANTS = [];
    for (let i = 0; i < 5; i++) {
        ROUND_CONSTANTS.push([
            `0x${(i * 2 + 1).toString(16).padStart(64, '0')}`,
            `0x${(i * 2 + 2).toString(16).padStart(64, '0')}`
        ]);
    }
    
    // 生成 Circom 参数文件
    const paramsContent = `pragma circom 2.1.6;

// 自动生成的 Poseidon2 参数
const ROUND_CONSTANTS = [
${ROUND_CONSTANTS.map(rc => `    [${rc.join(', ')}]`).join(',\n')}
];`;
    
    // 保存参数文件
    const paramsPath = path.join(__dirname, '../circuits/poseidon2_params.circom');
    fs.writeFileSync(paramsPath, paramsContent);
    
    console.log('Poseidon2 参数已生成:', paramsPath);
    
    // 生成测试输入文件
    const testInput = {
        message: "123456789"
    };
    const testInputPath = path.join(__dirname, '../test/input.json');
    fs.writeFileSync(testInputPath, JSON.stringify(testInput, null, 2));
    console.log('测试输入文件已生成:', testInputPath);
}

generatePoseidon2Params();
