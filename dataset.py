import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ThroatDataset(Dataset):
    def __init__(self, data_dir, max_len=6000):
        """
        data_dir: 存放csv数据的文件夹路径
        max_len: 统一设定的最大长度（比如 3 秒 * 2000Hz = 6000点）
        """
        self.file_list = glob.glob(os.path.join(data_dir, "*.csv"))
        self.max_len = max_len

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # 从文件名解析标签 (例如 "data/0_sample45.csv" 解析出标签 0)
        basename = os.path.basename(file_path)
        label = int(basename.split('_')[0])
        
        # 读取 CSV 里的电压数据
        data = np.loadtxt(file_path, delimiter=",")
        
        # --- 核心预处理：长度归一化 ---
        if len(data) > self.max_len:
            data = data[:self.max_len] # 太长就截断
        elif len(data) < self.max_len:
            # 太短就在后面补 0 (Padding)
            pad_width = self.max_len - len(data)
            data = np.pad(data, (0, pad_width), 'constant', constant_values=(0, 0))
            
        # 转换为 PyTorch 需要的 Tensor 格式: [通道数(1), 长度]
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return data_tensor, label_tensor

# 测试一下能不能正常读取
if __name__ == "__main__":
    dataset = ThroatDataset("data")
    print(f"总共找到 {len(dataset)} 个样本")
    sample_data, sample_label = dataset[0]
    print(f"第一个样本的数据维度: {sample_data.shape}, 标签: {sample_label}")