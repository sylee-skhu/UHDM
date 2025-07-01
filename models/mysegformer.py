import torch.nn as nn
from .segformer import SegFormer
from .utils import PatchDiscriminator


class MySegFormer(nn.Module):
    def __init__(self, num_out_channels=3, model_name="mit_b0", num_features=64):
        super().__init__()
        self.G = SegFormer(num_out_channels=num_out_channels, model_name=model_name)
        self.D = PatchDiscriminator(
            num_features=num_features,
            num_layers=3
        )

    def forward(self, data, mode="G"):
        outputs = self.G(data)
        out_full = outputs[0]
        real = data['label']
        D_real = self.D(real)
        D_fake = self.D(out_full)
        return outputs + [D_real, D_fake]
