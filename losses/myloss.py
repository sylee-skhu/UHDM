import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True, feature_layers=[0, 1, 2, 3]):
        super().__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.resize = resize
        self.feature_layers = feature_layers

    def forward(self, input, target):
        device = input.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        input = (input - mean) / std
        target = (target - mean) / std
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in self.feature_layers:
                loss += F.l1_loss(x, y)
        return loss


class MyLoss(nn.Module):
    """
    outputs: [out_full, out_half, out_quarter]
    targets: gt_full
    adversarial_loss: (optional) Hinge Loss 버전
    """

    def __init__(self, lam_p=1.0, lam_adv=0.01):
        super().__init__()
        self.vgg_loss = VGGPerceptualLoss()
        self.mse = nn.MSELoss()
        self.lam_p = lam_p
        self.lam_adv = lam_adv

    def forward(self, outputs, gt_full, d_loss=False):
        out_full, out_half, out_quarter, D_real, D_fake = outputs

        if d_loss:
            # Hinge loss (Discriminator, for reference)
            total_loss = F.relu(1. - D_real).mean() + F.relu(1. + D_fake).mean()

        else:
            gt_half = F.interpolate(gt_full, scale_factor=0.5, mode='bilinear', align_corners=False)
            gt_quarter = F.interpolate(gt_full, scale_factor=0.25, mode='bilinear', align_corners=False)

            # Pixel + Perceptual loss
            loss_full = self.mse(out_full, gt_full) + self.lam_p * self.vgg_loss(out_full, gt_full)
            loss_half = self.mse(out_half, gt_half) + self.lam_p * self.vgg_loss(out_half, gt_half)
            loss_quarter = self.mse(out_quarter, gt_quarter) + self.lam_p * self.vgg_loss(out_quarter, gt_quarter)
            pixel_perc_loss = loss_full + loss_half + loss_quarter

            # Hinge adversarial loss (Generator)
            adv_loss = -D_fake.mean()

            total_loss = pixel_perc_loss + self.lam_adv * adv_loss

        return total_loss
