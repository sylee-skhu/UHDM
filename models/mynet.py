import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, num_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf = num_features
        for _ in range(1, num_layers):
            layers += [
                nn.Conv2d(nf, nf*2, 4, stride=2, padding=1),
                nn.BatchNorm2d(nf*2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            nf *= 2
        layers += [
            nn.Conv2d(nf, 1, 4, padding=1)
        ]
        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class MyNet(nn.Module):
    def __init__(self, num_channels=3, num_features=64):
        super().__init__()
        self.G = UNet(
            num_channels=num_channels,
            num_features=num_features,
            num_out_channels=num_channels
        )
        self.D = PatchDiscriminator(
            in_channels=num_channels,
            num_features=num_features,
            num_layers=3
        )
        self._initialize_weights()

    def forward(self, data, mode="G"):
        x = data['in_img']
        outputs = self.G(x)
        x = data['in_img']
        real = data['label']
        outs = self.G({'in_img': x})  # [out_full, out_half, out_quarter]
        out_full = outs[0]

        D_real = self.D(real)
        D_fake = self.D(out_full)
        return outs + [D_real, D_fake]  # [out_full, out_half, out_quarter, D_real, D_fake]

    def _initialize_weights(self):
        pass
