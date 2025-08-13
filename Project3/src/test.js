# test/test.js

const { wasm } = require("circom_tester");
const chai = require("chai");
const path = require("path");
const fs = require("fs");
const assert = chai.assert;

describe("Poseidon2 电路测试", function () {
    this.timeout(100000);
    
    let circuit;
    
    before(async () => {
        // 编译电路
        circuit = await wasm(path.join(__dirname, "../circuits/poseidon2.circom"));
    });
    
    it("应正确计算 Poseidon2 哈希", async () => {
        const input = JSON.parse(fs.readFileSync(path.join(__dirname, "input.json")));
        
        // 计算见证
        const witness = await circuit.calculateWitness(input, true);
        
        // 获取输出
        const output = witness[1];
        
        // 预期输出 (基于示例实现)
        const expectedOutput = "1787791386975108476636143458450883075490";
        
        // 验证输出
        assert.equal(output.toString(), expectedOutput, "哈希值不匹配");
        
        console.log("测试通过! 输入:", input.message);
        console.log("输出哈希:", output.toString());
    });
});
