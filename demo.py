from datetime import datetime
import logging
import lpips
import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision
import random
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
import os
from models import create_model
from datasets import create_dataset
from tqdm import tqdm
from utils.common import *
from config.config import args
from PIL import Image
from PIL import ImageFile


def demo_step(args, data, model, save_path, device):
    """
    - args: config/arguments
    - data: DataLoader에서 받은 {'in_img': tensor, 'number': filename 등}
    - model: 학습된 모델 (eval 모드로 호출)
    - save_path: 결과 저장 경로
    - device: 'cuda' 또는 'cpu'
    """
    model.eval()
    in_img = data['in_img']
    number = data['number']
    b, c, h, w = in_img.size()

    # pad image such that the resolution is a multiple of 32
    w_pad = (math.ceil(w/32)*32 - w) // 2
    h_pad = (math.ceil(h/32)*32 - h) // 2
    w_odd_pad = w_pad
    h_odd_pad = h_pad
    if w % 2 == 1:
        w_odd_pad += 1
    if h % 2 == 1:
        h_odd_pad += 1

    # shallow copy for safety
    in_img_pad = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
    data_mod = dict(data)
    data_mod['in_img'] = in_img_pad

    with torch.no_grad():
        out_1 = model(data_mod)[0]
        # crop padding
        if h_pad != 0:
            out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
        if w_pad != 0:
            out_1 = out_1[:, :, :, w_pad:-w_odd_pad]

    # save output image
    if args.SAVE_IMG:
        out_save = out_1.detach().cpu()
        # number은 이미지 이름/번호. list인지 str인지 확인!
        if isinstance(number, (list, tuple)):
            filename = number[0]
        else:
            filename = number
        save_ext = getattr(args, "SAVE_IMG", "png")
        out_file = os.path.join(save_path, f'test_{filename}.{save_ext}')
        torchvision.utils.save_image(out_save, out_file)


def demo(args, TestImgLoader, model, save_path, device):
    tbar = tqdm(TestImgLoader)
    for batch_idx, data in enumerate(tbar):
        model.eval()
        demo_step(args, data, model, save_path, device)   # <-- 변경!
        desc = 'Test demo'
        tbar.set_description(desc)
        tbar.update()


def init():
    args.TEST_RESULT_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'test_result')
    mkdir(args.TEST_RESULT_DIR)
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.GPU_ID
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.SEED, deterministic=False)
    return device


def load_checkpoint(model):
    if args.LOAD_PATH:
        load_path = args.LOAD_PATH
        save_path = args.TEST_RESULT_DIR + '/customer'
        log_path = args.TEST_RESULT_DIR + '/customer_result.log'
    else:
        print('Please specify a checkpoint path in the config file!!!')
        raise NotImplementedError
    mkdir(save_path)
    if load_path.endswith('.pth'):
        model_state_dict = torch.load(load_path)
    else:
        model_state_dict = torch.load(load_path)['state_dict']
    model.load_state_dict(model_state_dict)
    return load_path, save_path, log_path


def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def main():
    device = init()
    # load model
    model = create_model(args).to(device)

    # load checkpoint
    load_path, save_path, log_path = load_checkpoint(model)

    # set logging for recording information or metrics
    set_logging(log_path)
    logging.warning(datetime.now())
    logging.warning('load model from %s' % load_path)
    logging.warning('save image results to %s' % save_path)
    logging.warning('save logger to %s' % log_path)

    # Create dataset
    test_path = args.DEMO_DATASET
    args.BATCH_SIZE = 1
    DemoImgLoader = create_dataset(args, data_path=test_path, mode='demo')

    # test demo
    demo(args, DemoImgLoader, model, save_path, device)


if __name__ == '__main__':
    main()
