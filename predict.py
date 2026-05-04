import torch
import numpy as np
from model import PureCNNThroat

# 1. 定义你的词表（跟训练时的标签 0, 1, 2 一一对应）
vocab = {
    0: "好",
    1: "我饿了",
    2: "请帮帮我"
}

def predict_new_signal(csv_file_path):
    print(f"正在分析波形文件: {csv_file_path} ...")
    
    # 2. 实例化模型，并加载我们刚刚训练好的“大脑”
    model = PureCNNThroat(num_classes=3)
    # 加载权重
    model.load_state_dict(torch.load("throat_model_weights.pth"))
    model.eval() # 切换到测试/评估模式（非常重要！）
    
    # 3. 读取新的硬件采集数据
    data = np.loadtxt(csv_file_path, delimiter=",")
    
    # 4. 长度预处理（跟 dataset.py 里一模一样）
    max_len = 6000
    if len(data) > max_len:
        data = data[:max_len]
    elif len(data) < max_len:
        data = np.pad(data, (0, max_len - len(data)), 'constant')
        
    # 转换为 Tensor，并增加 Batch 和 Channel 维度 -> [1, 1, 6000]
    tensor_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 5. 送入模型进行预测
    with torch.no_grad(): # 预测时不需要计算梯度，省内存提速度
        output = model(tensor_data)
        
        # output 输出的是三个类别的打分，分数最高的那个就是模型的最终答案
        _, predicted_label = torch.max(output.data, 1)
        label_index = predicted_label.item()
        
    print("=====================================")
    print(f"🎙️ AI 识别结果: 【 {vocab[label_index]} 】")
    print("=====================================\n")

if __name__ == "__main__":
    # 我们从 data 文件夹里随便挑一个文件来模拟“实时采集的新数据”
    # 你可以把这里换成任意一个你想测试的 csv 文件路径
    test_file = "data/2_sample42.csv" 
    
    predict_new_signal(test_file)