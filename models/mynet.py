import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, num_channels=3, num_features=64):
        super().__init__()
        self._initialize_weights()

    def forward(self, data):
        x = data['in_img']
        y = data['label']
        return [x]

    def _initialize_weights(self):
        pass
