import os
import numpy as np
import pandas as np_pd

# 假设我们有 3 句话要识别，分别打上标签 0, 1, 2
# 标签0："好" ； 标签1："我饿了" ； 标签2："请帮帮我"
os.makedirs('data', exist_ok=True)

print("开始生成模拟数据...")
for label in range(3):
    for sample_id in range(100): # 每句话生成 100 个样本
        # 模拟不同语速：长度在 2000 到 4000 之间随机
        length = np.random.randint(2000, 4000) 
        
        # 制造不同类别的特征差异 (为了让模型能学到东西)
        if label == 0:   noise = np.sin(np.linspace(0, 10, length)) + np.random.randn(length)*0.1
        elif label == 1: noise = np.cos(np.linspace(0, 20, length)) + np.random.randn(length)*0.2
        else:            noise = np.sin(np.linspace(0, 30, length)) * np.cos(np.linspace(0, 5, length)) + np.random.randn(length)*0.1
        
        # 保存为 CSV 文件，命名规则：标签_样本号.csv (例如: 0_sample45.csv)
        filename = f"data/{label}_sample{sample_id}.csv"
        # 假设单片机传上来的是只有一列电压值的数据
        np.savetxt(filename, noise, delimiter=",", fmt="%.4f")

print("成功生成 300 个模拟数据文件，存放在 data/ 目录下！")