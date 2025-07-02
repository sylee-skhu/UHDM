import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet3plus import UNet3Plus
from .utils import PatchDiscriminator


class MyUNet3Plus(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = UNet3Plus(
            num_classes=3,
            deep_sup=True
        )
        self.D = PatchDiscriminator(
            num_features=64,
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
