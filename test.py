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
import logging


def test_step(args, data, model, device, save_path, compute_metrics):
    """
    테스트 한 step (forward + metric 계산 + 결과 저장)
    """
    number = data['number']
    in_img = data['in_img']
    label = data['label']
    b, c, h, w = in_img.size()
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
        st = time.time()
        out_1 = model(data_mod)[0]
        cur_time = time.time()-st
        if h_pad != 0:
            out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
        if w_pad != 0:
            out_1 = out_1[:, :, :, w_pad:-w_odd_pad]
    # metric 계산
    cur_lpips, cur_psnr, cur_ssim = 0.0, 0.0, 0.0
    if args.EVALUATION_METRIC:
        cur_lpips, cur_psnr, cur_ssim = compute_metrics.compute(out_1, label)
    # save images
    if args.SAVE_IMG:
        out_save = out_1.detach().cpu()
        torchvision.utils.save_image(
            out_save,
            save_path + '/' + 'test_%s' % number[0] + '.%s' % args.SAVE_IMG
        )
    return cur_psnr, cur_ssim, cur_lpips, cur_time


def test(args, TestImgLoader, model, device, save_path, compute_metrics):
    tbar = tqdm(TestImgLoader)
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_time = 0
    avg_val_time = 0
    for batch_idx, data in enumerate(tbar):
        model.eval()
        cur_psnr, cur_ssim, cur_lpips, cur_time = test_step(args, data, model, device, save_path, compute_metrics)
        number = data['number']
        if args.EVALUATION_METRIC:
            logging.info('%s: LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f' % (number[0], cur_lpips, cur_psnr, cur_ssim))
        if args.EVALUATION_TIME:
            logging.info('%s: TIME is %.4f' % (number[0], cur_time))
        total_psnr += cur_psnr
        avg_val_psnr = total_psnr / (batch_idx+1)
        total_ssim += cur_ssim
        avg_val_ssim = total_ssim / (batch_idx+1)
        total_lpips += cur_lpips
        avg_val_lpips = total_lpips / (batch_idx+1)
        if batch_idx > 5:
            total_time += cur_time
            avg_val_time = total_time / (batch_idx-5)
        if args.EVALUATION_METRIC:
            desc = 'Test: Avg. LPIPS = %.4f, Avg. PSNR = %.4f and SSIM = %.4f' % (avg_val_lpips, avg_val_psnr, avg_val_ssim)
        elif args.EVALUATION_TIME:
            desc = 'Avg. TIME is %.4f' % avg_val_time
        else:
            desc = 'Test without any evaluation'
        tbar.set_description(desc)
        tbar.update()
    if args.EVALUATION_METRIC:
        logging.warning('Avg. LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f' % (avg_val_lpips, avg_val_psnr, avg_val_ssim))
    if args.EVALUATION_TIME:
        logging.warning('Avg. TIME is %.4f' % avg_val_time)


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
        load_epoch = args.TEST_EPOCH
        if load_epoch == 'auto':
            load_path = args.NETS_DIR + '/checkpoint_latest.tar'
            save_path = args.TEST_RESULT_DIR + '/latest'
            log_path = args.TEST_RESULT_DIR + '/latest_result.log'
        else:
            load_path = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
            save_path = args.TEST_RESULT_DIR + '/' + '%04d' % load_epoch
            log_path = args.TEST_RESULT_DIR + '/%04d_' % load_epoch + 'result.log'
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

    # computational cost for the model
    if args.EVALUATION_COST:
        calculate_cost(model, input_size=(1, 3, 2176, 3840))

    logging.warning('load model from %s' % load_path)
    logging.warning('save image results to %s' % save_path)
    logging.warning('save logger to %s' % log_path)

    compute_metrics = None
    if args.EVALUATION_TIME:
        args.EVALUATION_METRIC = False
    if args.EVALUATION_METRIC:
        from utils.metric import create_metrics
        compute_metrics = create_metrics(args, device=device)

    # Create dataset
    test_path = args.TEST_DATASET
    args.BATCH_SIZE = 1
    TestImgLoader = create_dataset(args, data_path=test_path, mode='test')

    # test (이제 model_fn_test 대신 test_step!)
    test(args, TestImgLoader, model, device, save_path, compute_metrics)


if __name__ == '__main__':
    main()
