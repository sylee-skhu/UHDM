import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet
from .utils import PatchDiscriminator


class MyUNet(nn.Module):
    def __init__(self, num_channels=3, num_features=64):
        super().__init__()
        self.G = UNet(
            num_channels=num_channels,
            num_features=num_features,
            num_out_channels=num_channels
        )
        self.D = PatchDiscriminator(
            num_features=num_features,
            num_layers=3
        )
        self._initialize_weights()

    def forward(self, data, mode="G"):
        outputs = self.G(data)
        out_full = outputs[0]
        real = data['label']
        D_real = self.D(real)
        D_fake = self.D(out_full)
        return outputs + [D_real, D_fake]  # [out_full, out_half, out_quarter, D_real, D_fake]

    def _initialize_weights(self):
        pass
