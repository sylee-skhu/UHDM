import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerDecodeHead, SegformerConfig
from .utils import FreqEdgeFusionBlock, FreqEdgeFusionBlockV2
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale ** 2), 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)


class PixelRegressionHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, upsample_scales=(4, 8, 16, 32)):
        super().__init__()
        self.upsample_modules = nn.ModuleList([
            PixelShuffleUpsample(in_ch, 32, scale)
            for in_ch, scale in zip(in_channels, upsample_scales)
        ])
        self.out_conv = nn.Conv2d(32 * len(in_channels), out_channels, 1)
        self._initialize_weights()

    def forward(self, feats, out_size):
        up_feats = [self.upsample_modules[i](f) for i, f in enumerate(feats)]
        x = torch.cat(up_feats, dim=1)
        out = self.out_conv(x)
        # shape 미세 불일치 보정
        if out.shape[-2:] != out_size:
            out = F.interpolate(out, size=out_size, mode='bilinear', align_corners=False)
        return out

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.out_conv.weight, nonlinearity='relu')
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)


class SegFormer(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_out_channels=3):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(model_name, output_hidden_states=True)
        self.freq_blocks = nn.ModuleList([FreqEdgeFusionBlockV2(dim) for dim in self.encoder.config.hidden_sizes])
        self.head = PixelRegressionHead(self.encoder.config.hidden_sizes, out_channels=num_out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data['in_img']  # (B, 3, H, W)
        H, W = x.shape[2], x.shape[3]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        feats = self.encoder(x, output_hidden_states=True).hidden_states
        feats = [block(f) for block, f in zip(self.freq_blocks, feats)]
        out = self.head(feats, out_size=(H, W))
        out = self.sigmoid(out)  # Normalize output to [0, 1]
        return [out]
