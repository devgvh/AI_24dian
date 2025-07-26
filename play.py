#!/usr/bin/env python3
from transformers import pipeline
import sympy
import json
from datasets import Dataset
import torch

# 检查是否有GPU可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

solver = pipeline("text2text-generation",
                  model="./24points_rpn_nosol",
                  tokenizer="./24points_rpn_nosol",
                  device=device)
# 运行单个测试用例
prompt = "solve: 1 5 5 5"
out = solver(prompt)
print('prompt:', prompt)
print(out)


def evaluate_rpn(expression):
    """计算后缀表达式的值"""
    if expression == "<no_solution>":
        return None
    
    stack = []
    tokens = expression.split()
    
    for token in tokens:
        if token.isdigit():
            stack.append(float(token))
        elif token in ['+', '-', '*', '/']:
            if len(stack) < 2:
                return None
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                if b == 0:
                    return None
                stack.append(a / b)
        else:
            return None
    
    return stack[0] if len(stack) == 1 else None

def test_all_cases():
    """测试所有测试用例并计算成功率"""
    # 读取所有测试用例
    test_cases = []
    with open('val.jsonl', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                test_cases.append({
                    'line': line_num,
                    'input': data['input'],
                    'prompt': f"solve: {data['input']}",
                    'expected_output': data['output']  # 保留期望输出用于验证无解情况
                })
            except Exception as e:
                print(f"第 {line_num} 行解析错误: {e}")
                continue
    
    print(f"总共读取了 {len(test_cases)} 个测试用例")
    
    # 创建数据集
    dataset = Dataset.from_list(test_cases)
    
    # 批量生成答案
    print("开始批量生成答案...")
    results = solver(list(dataset['prompt']), batch_size=64)
    
    # 验证结果
    correct = 0
    failed_cases = []
    
    for i, (test_case, result) in enumerate(zip(test_cases, results)):
        generated_output = result['generated_text'].strip()
        generated_result = evaluate_rpn(generated_output)
        expected_result = evaluate_rpn(test_case['expected_output'])
        
        # 检查结果是否正确
        is_correct = False
        
        if expected_result is None:  # 期望无解
            if generated_result is None:  # 生成也是无解
                is_correct = True
        else:  # 期望有解
            if generated_result is not None and abs(generated_result - 24) < 1e-6:  # 生成有解且等于24
                is_correct = True
        
        if is_correct:
            correct += 1
        else:
            failed_cases.append({
                'line': test_case['line'],
                'input': test_case['input'],
                'expected': test_case['expected_output'],
                'generated': generated_output,
                'expected_result': expected_result,
                'generated_result': generated_result
            })
        
        # 每100个测试用例打印一次进度
        if (i + 1) % 100 == 0:
            print(f"已测试 {i + 1} 个用例，当前成功率: {correct/(i+1)*100:.2f}%")
    
    # 打印最终结果
    total = len(test_cases)
    success_rate = correct / total * 100 if total > 0 else 0
    print(f"\n=== 测试结果 ===")
    print(f"总测试用例数: {total}")
    print(f"正确用例数: {correct}")
    print(f"成功率: {success_rate:.2f}%")
    
    # 打印前10个失败的用例
    if failed_cases:
        print(f"\n全部失败的用例:")
        for i, case in enumerate(failed_cases):
            print(f"{i+1}. 输入: {case['input']}")
            print(f"   期望: {case['expected']} (结果: {case['expected_result']})")
            print(f"   生成: {case['generated']} (结果: {case['generated_result']})")
            print()



'''
print("\n" + "="*50)
print("开始批量测试...")

# 运行所有测试用例
test_all_cases()
'''

