import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, num_channels=3, num_features=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_features, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_features, num_channels, 3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        # out_1, out_2, out_3를 모두 반환하도록 기존 포맷 맞춤
        return x, x, x  # 필요시 원하는 방식대로 수정

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
