import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import MultiGaussianDiffFusion
from .utils import FreqEdgeFusionBlock


class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def forward(self, x):
        return self.block(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, enc_feat):
        x = self.up(x)
        # Handle possible shape mismatch due to odd input sizes
        if x.shape[-2:] != enc_feat.shape[-2:]:
            dh = enc_feat.shape[-2] - x.shape[-2]
            dw = enc_feat.shape[-1] - x.shape[-1]
            enc_feat = enc_feat[..., dh//2:enc_feat.shape[-2]-dh+dh//2, dw//2:enc_feat.shape[-1]-dw+dw//2]
        x = torch.cat([x, enc_feat], dim=1)
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, num_channels=3, num_features=64, num_out_channels=3):
        super().__init__()
        # Encoder
        self.enc1 = UNetEncoderBlock(num_channels, num_features)
        self.enc2 = UNetEncoderBlock(num_features, num_features * 2)
        self.enc3 = UNetEncoderBlock(num_features * 2, num_features * 4)
        self.enc4 = UNetEncoderBlock(num_features * 4, num_features * 8)
        self.pool = nn.MaxPool2d(2)

        # MultiGaussianDiffFusion for feature fusion
        self.skip1 = MultiGaussianDiffFusion(num_features)
        self.skip2 = MultiGaussianDiffFusion(num_features * 2)
        self.skip3 = MultiGaussianDiffFusion(num_features * 4)
        self.skip4 = MultiGaussianDiffFusion(num_features * 8)

        # Bottleneck
        # self.bottleneck = UNetEncoderBlock(num_features * 8, num_features * 16)
        self.bottleneck = FreqEdgeFusionBlock(num_features * 8, num_features * 16)

        # Decoder
        self.dec4 = UNetDecoderBlock(num_features * 16, num_features * 8)
        self.dec3 = UNetDecoderBlock(num_features * 8, num_features * 4)
        self.dec2 = UNetDecoderBlock(num_features * 4, num_features * 2)
        self.dec1 = UNetDecoderBlock(num_features * 2, num_features)

        # Output layers for each scale
        self.out_full = nn.Conv2d(num_features, num_out_channels, 1)
        self.out_half = nn.Conv2d(num_features * 2, num_out_channels, 1)
        self.out_quarter = nn.Conv2d(num_features * 4, num_out_channels, 1)
        self._initialize_weights()

    def forward(self, data):
        x = data['in_img']
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        # Skip connections with MultiGaussianDiffFusion
        e1 = self.skip1(e1)
        e2 = self.skip2(e2)
        e3 = self.skip3(e3)
        e4 = self.skip4(e4)
        # Decoder with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        # Outputs at three scales
        out_full = self.out_full(d1)
        out_half = self.out_half(d2)
        out_quarter = self.out_quarter(d3)
        return [out_full, out_half, out_quarter]

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.out_full.weight)
        nn.init.zeros_(self.out_full.bias)
        nn.init.kaiming_normal_(self.out_half.weight)
        nn.init.zeros_(self.out_half.bias)
        nn.init.kaiming_normal_(self.out_quarter.weight)
        nn.init.zeros_(self.out_quarter.bias)
