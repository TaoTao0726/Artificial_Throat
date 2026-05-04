import torch
import torch.nn as nn

class PureCNNThroat(nn.Module):
    def __init__(self, num_classes=3):
        super(PureCNNThroat, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4), 
            
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.AdaptiveMaxPool1d(1) # 全局池化魔法
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x) 
        x = x.view(x.size(0), -1) 
        out = self.classifier(x)
        return out