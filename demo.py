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
from models import create_model, demo_step         # <-- 변경!
from datasets.utils import default_toTensor              # <-- util import
from tqdm import tqdm
from utils.loss_util import *
from utils.common import *
from config.config import args
from PIL import Image
from PIL import ImageFile


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
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
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
    model._initialize_weights()

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
