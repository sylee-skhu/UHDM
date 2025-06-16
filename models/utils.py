import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_positional_encoding(batch, height, width, device=None, dtype=None):
    # x, y normalized coordinates [0, 1]
    y_embed = torch.linspace(0, 1, steps=height, device=device, dtype=dtype).view(1, height, 1).expand(batch, height, width)
    x_embed = torch.linspace(0, 1, steps=width, device=device, dtype=dtype).view(1, 1, width).expand(batch, height, width)
    pos = torch.stack([y_embed, x_embed], dim=1)  # (B, 2, H, W)
    return pos


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


def get_gaussian_kernel(kernel_size, channels):
    # 1D 가우시안 커널을 생성하는 함수 (OpenCV)
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    # 1D 가우시안 생성
    grid = torch.arange(kernel_size).float() - kernel_size // 2
    kernel_1d = torch.exp(-(grid ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    # 2D 가우시안 커널 만들기 (채널별 적용)
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    return kernel_2d


class MultiGaussianDiffFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Fixed Gaussian kernels for conv2d
        self.channels = channels
        self.register_buffer('kernel3', get_gaussian_kernel(3, channels))
        self.register_buffer('kernel5', get_gaussian_kernel(5, channels))
        self.register_buffer('kernel7', get_gaussian_kernel(7, channels))
        # Trainable scalar weights
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.w3 = nn.Parameter(torch.tensor(1.0))
        self.w4 = nn.Parameter(torch.tensor(1.0))

    def forward(self, s):
        # 각 채널별로 depthwise conv
        s3 = F.conv2d(s, self.kernel3, padding=1, groups=self.channels)
        s5 = F.conv2d(s, self.kernel5, padding=2, groups=self.channels)
        s7 = F.conv2d(s, self.kernel7, padding=3, groups=self.channels)
        s3_diff = s - s3
        s5_diff = s3 - s5
        s7_diff = s5 - s7
        s_out = self.w1 * s + self.w2 * s3_diff + self.w3 * s5_diff + self.w4 * s7_diff
        return s_out


class FrequencyDomainModulator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Learnable modulation scalars
        self.alpha_low = nn.Parameter(torch.tensor(0.5))
        self.alpha_high = nn.Parameter(torch.tensor(0.5))

        # Frequency-domain 1×1 convs operate on real+imag concatenated as 2C channels
        self.freq_conv1 = nn.Conv2d(2*channels, 2*channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.freq_conv2 = nn.Conv2d(2*channels, 2*channels, kernel_size=1)

        # Spatial-domain 1×1 conv after inverse FFT
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # He initialization for 1x1 convs
        nn.init.kaiming_normal_(self.freq_conv1.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.freq_conv2.weight, mode='fan_in')
        nn.init.kaiming_normal_(self.spatial_conv.weight, mode='fan_in')
        if self.freq_conv1.bias is not None:
            nn.init.zeros_(self.freq_conv1.bias)
        if self.freq_conv2.bias is not None:
            nn.init.zeros_(self.freq_conv2.bias)
        if self.spatial_conv.bias is not None:
            nn.init.zeros_(self.spatial_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B×C×H×W
        # 1. FFT to frequency domain
        f = torch.fft.fft2(x, norm='ortho')  # complex tensor B×C×H×W

        # 2. Magnitude spectrum
        m = torch.abs(f)
        # 3. Mean amplitude spectrum
        m_mean = m.mean(dim=[-2, -1], keepdim=True)

        # 4. Binary indicator masks (0 or 1)
        mask_high = (m > m_mean).type_as(m)               # 1 where m > m_mean, else 0
        mask_low = (m <= 0.5 * m_mean).type_as(m)        # 1 where m <= 0.5*m_mean, else 0

        # 5. Modulate f with learnable scales
        scale = 1 + self.alpha_low * mask_low + self.alpha_high * mask_high
        f2 = f * scale

        # 6. Frequency-domain convs: split into real & imag channels
        real = f2.real
        imag = f2.imag
        freq_in = torch.cat([real, imag], dim=1)  # B×2C×H×W
        z = self.freq_conv1(freq_in)
        z = self.gelu(z)
        z = self.freq_conv2(z)

        # Reconstruct complex tensor for inverse FFT
        real_z, imag_z = z.chunk(2, dim=1)
        f2_processed = torch.complex(real_z, imag_z)

        # 7. Inverse FFT back to spatial domain
        y = torch.fft.ifft2(f2_processed, norm='ortho').real  # B×C×H×W

        # 8. Spatial conv + sigmoid activation
        f3 = torch.sigmoid(self.spatial_conv(y))

        # 9. Modulate original input
        out = x * f3
        return out


class SobelEdgeAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # 3x3 Conv 적용
        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # 8방향 Sobel 커널 (오리지널 4방향 + 45/135도 대각선 2방향 + 역방향 2개)
        kernels = torch.tensor([
            [[-1, 0, 1],   # 0도 (수평)
             [-2, 0, 2],
             [-1, 0, 1]],

            [[-1, -2, -1],  # 90도 (수직)
             [0,  0,  0],
             [1,  2,  1]],

            [[-2, -1, 0],  # 45도
             [-1, 0, 1],
             [0, 1, 2]],

            [[0, 1, 2],   # 135도
             [-1, 0, 1],
             [-2, -1, 0]],
        ], dtype=torch.float32)  # (4, 3, 3)

        # (4, 1, 3, 3) → (channels*4, 1, 3, 3)
        kernels = kernels.unsqueeze(1)
        self.kernels = nn.Parameter(
            kernels.repeat(channels, 1, 1, 1), requires_grad=False
        )

        # 1x1 conv for attention
        self.conv1x1 = nn.Conv2d(1, 1, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.pre_conv.weight, mode='fan_out')
        if self.pre_conv.bias is not None:
            nn.init.zeros_(self.pre_conv.bias)
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out')
        if self.conv1x1.bias is not None:
            nn.init.zeros_(self.conv1x1.bias)

    def forward(self, x):
        # 1. 3x3 Conv 적용
        x_conv = self.pre_conv(x)

        # 2. 4방향 Sobel
        # (B, C, H, W) → (B, C*4, H, W) 각 채널마다 4개의 Sobel 필터 적용
        edge_maps = F.conv2d(
            x_conv, self.kernels, bias=None, stride=1, padding=1,
            groups=self.channels
        )  # shape: (B, C*4, H, W)

        # 3. 각 채널별 4방향 gradient에서 abs max 취함
        edge_maps = edge_maps.view(x.size(0), self.channels, 4, x.size(2), x.size(3))
        x_edge = edge_maps.abs().max(dim=2)[0]  # (B, C, H, W)

        # 4. 채널 평균 & 최대값 풀링
        avg_pool = x_edge.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        max_pool = x_edge.max(dim=1, keepdim=True)[0]     # (B, 1, H, W)
        M = avg_pool + max_pool                           # (B, 1, H, W)

        # 5. 1x1 conv → sigmoid → attention map
        M2 = torch.sigmoid(self.conv1x1(M))               # (B, 1, H, W)

        # 6. Attention 적용
        out = x * M2                                      # (B, C, H, W)
        return out


class FreqEdgeFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 인코더 부분
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 주파수/엣지 모듈
        self.freq_mod = FrequencyDomainModulator(out_channels)
        self.edge_att = SobelEdgeAttention(out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        # block의 Conv2d 초기화
        for m in self.block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 1. 인코더
        x_encoded = self.block(x)

        # 2. 주파수/엣지 모듈
        x_freq = self.freq_mod(x_encoded)
        x_edge = self.edge_att(x_encoded)

        # 3. Residual 연결
        out = x_encoded * x_freq * x_edge + x_encoded
        return out
