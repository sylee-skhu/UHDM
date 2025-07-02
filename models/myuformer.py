import torch
import torch.nn as nn
import torch.nn.functional as F
from .uformer import Uformer
from .utils import PatchDiscriminator


class MyUFormer(nn.Module):
    def __init__(self, arch='Uformer_B'):
        super().__init__()
        self.G = get_arch(arch)
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

def get_arch(arch):

    if arch == 'Uformer_T':
        model_restoration = Uformer(img_size=256,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    elif arch == 'Uformer_S':
        model_restoration = Uformer(img_size=256,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
    elif arch == 'Uformer_B':
        model_restoration = Uformer(img_size=256,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff', depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True)  
    else:
        raise Exception("Arch error!")

    return model_restoration