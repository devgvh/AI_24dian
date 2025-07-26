#!/usr/bin/env python3
"""
生成 24 点数据集（含无解）
- 后缀表达式
- 剪枝 DFS
- 多进程并行
"""
import itertools, json, random, math, os
from fractions import Fraction
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

OPS = {'+': lambda a, b: a + b,
       '-': lambda a, b: a - b,
       '*': lambda a, b: a * b,
       '/': lambda a, b: a / b if b != 0 else None}

CUT_FACTOR = 2 ** 20  # 允许的最大中间值，防爆炸


def dfs(nums, path):
    """
    回溯生成所有合法后缀式（剪枝版）
    nums  : list[Fraction] 剩余数字
    path  : list[str]      已生成的 token
    """
    if len(nums) == 1:
        if abs(nums[0] - 24) < 1e-9:
            yield " ".join(path)
        return

    # 剪枝：若剩余数字全部乘/加也达不到 24 则剪枝
    max_rest = max(abs(x) for x in nums)
    if max_rest * math.factorial(len(nums)) < 24 / CUT_FACTOR:
        return

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            a, b = nums[i], nums[j]
            rest = nums[:i] + nums[i + 1:j] + nums[j + 1:]
            for op, fn in OPS.items():
                if op == '/' and b == 0:
                    continue
                new_num = fn(a, b)
                if new_num is None or abs(new_num) > CUT_FACTOR:
                    continue
                yield from dfs(rest + [new_num],
                               path + [str(a), str(b), op])


def worker(quad):
    """单进程任务：四元组 -> list[dict]"""
    solved = []
    for perm in set(itertools.permutations(quad)):
        for expr in dfs(list(map(Fraction, perm)), []):
            solved.append({"input": " ".join(map(str, sorted(quad))),
                           "output": expr})
    return solved


def generate():
    # 715 种无序四元组
    quads = list(itertools.combinations_with_replacement(range(1, 14), 4))
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(worker, quads), total=len(quads), desc="proc"))

    # 合并所有有解样本
    data = [item for sub in results for item in sub]

    # 添加无解样本
    solved_keys = {d["input"] for d in data}
    for quad in quads:
        key = " ".join(map(str, sorted(quad)))
        if key not in solved_keys:
            data.append({"input": key, "output": "<no_solution>"})

    random.shuffle(data)
    n = len(data)
    splits = {"train": slice(0, int(0.98 * n)),
              "val": slice(int(0.98 * n), int(0.99 * n)),
              "test": slice(int(0.99 * n), None)}
    for split, s in splits.items():
        with open(f"{split}.jsonl", "w", encoding="utf-8") as f:
            for d in data[s]:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Total: {len(data)}  (solved: {len(solved_keys)}, unsolvable: {n-len(solved_keys)})")


if __name__ == "__main__":
    generate()