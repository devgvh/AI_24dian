#!/usr/bin/env python3
"""
根据 trainer_state.json 画 loss 曲线
"""
import json, matplotlib.pyplot as plt
import pandas as pd
import os, glob

# 1. 找到任意一个 checkpoint 里的 trainer_state.json
state_file = glob.glob("**/trainer_state.json", recursive=True)[0]

# 2. 读取日志
with open(state_file) as f:
    data = json.load(f)
logs = pd.DataFrame(data["log_history"])

# 3. 过滤出带 loss 的行
train_logs = logs[logs["loss"].notnull()]
eval_logs  = logs[logs["eval_loss"].notnull()]

# 4. 画图
plt.figure(figsize=(8,4))
plt.plot(train_logs["step"], train_logs["loss"], label="train_loss")
if not eval_logs.empty:
    plt.plot(eval_logs["step"], eval_logs["eval_loss"], label="eval_loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()