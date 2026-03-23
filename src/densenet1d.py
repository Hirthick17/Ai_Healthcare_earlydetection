# src/densenet1d.py — Deep Learning 1D-CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, growth_rate * 4, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(growth_rate * 4)
        self.conv2 = nn.Conv1d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))

class DenseNet1D(nn.Module):
    # Upgraded to 20 sub-classes (1 Normal, 19 Abnormal Diagnoses)
    def __init__(self, in_channels=12, num_classes=20):
        super(DenseNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.db1 = DenseBlock(num_layers=4, in_channels=64, growth_rate=32)
        self.trans1 = Transition(in_channels=192, out_channels=96)
        
        self.db2 = DenseBlock(num_layers=4, in_channels=96, growth_rate=32)
        self.trans2 = Transition(in_channels=224, out_channels=112)
        
        self.classifier = nn.Sequential(
            nn.Linear(112, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.db1(out)
        out = self.trans1(out)
        out = self.db2(out)
        out = self.trans2(out)
        
        out = F.adaptive_avg_pool1d(out, 1).view(out.size(0), -1)
        logits = self.classifier(out)
        return logits
