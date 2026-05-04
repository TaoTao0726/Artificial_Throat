import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入我们刚刚写好的两个文件
from model import PureCNNThroat
from dataset import ThroatDataset

def train_model():
    # 1. 准备数据
    print("正在加载数据...")
    dataset = ThroatDataset("data", max_len=6000)
    # batch_size=32 表示每次抓取 32 句话一起训练，加快速度；shuffle=True 表示打乱顺序
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. 初始化模型、损失函数和优化器
    model = PureCNNThroat(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20 # 训练 20 轮
    
    print("开始训练...")
    for epoch in range(epochs):
        model.train() # 开启训练模式
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad() # 梯度清零
            
            # 前向传播
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播更新权重
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算一下当前批次的准确率
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
        # 打印每一轮的进度
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total_samples
        print(f"第 {epoch+1}/{epochs} 轮 - 误差(Loss): {avg_loss:.4f} - 准确率(Accuracy): {accuracy:.2f}%")
        
    # 3. 训练完成，保存模型参数（这样下次就能直接拿去识别，不用重训了）
    torch.save(model.state_dict(), "throat_model_weights.pth")
    print("训练完成！模型已保存为 throat_model_weights.pth")

if __name__ == "__main__":
    train_model()