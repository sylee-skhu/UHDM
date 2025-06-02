# dataset/dataset.py

import os
import torch
import torch.utils.data as data


def default_loader(path_set=[]):
    from PIL import Image
    from torchvision import transforms
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = transforms.ToTensor()(img)
        imgs.append(img)
    return imgs


def crop_loader(crop_size, x, y, path_set=[]):
    from PIL import Image
    from torchvision import transforms
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.crop((x, y, x + crop_size, y + crop_size))
        img = transforms.ToTensor()(img)
        imgs.append(img)
    return imgs


def resize_loader(resize_size, path_set=[]):
    from PIL import Image
    from torchvision import transforms
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.resize((resize_size, resize_size), Image.BICUBIC)
        img = transforms.ToTensor()(img)
        imgs.append(img)
    return imgs


def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)
    return composed_transform(img)
