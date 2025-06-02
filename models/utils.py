import torch
import torchvision
import time
import math
from utils.common import *


def train_step(args, data, model, loss_fn, device, iters):
    """
    한 step 훈련 (forward + loss + image 저장)
    """
    model.train()
    in_img = data['in_img'].to(device)
    label = data['label'].to(device)
    out_1, out_2, out_3 = model(in_img)
    loss = loss_fn(out_1, out_2, out_3, label)
    # save images
    if iters % args.SAVE_ITER == (args.SAVE_ITER - 1):
        in_save = in_img.detach().cpu()
        out_save = out_1.detach().cpu()
        gt_save = label.detach().cpu()
        res_save = torch.cat((in_save, out_save, gt_save), 3)
        save_number = (iters + 1) // args.SAVE_ITER
        torchvision.utils.save_image(
            res_save,
            args.VISUALS_DIR + '/visual_x%04d_' % args.SAVE_ITER + '%05d' % save_number + '.jpg'
        )
    return loss


def test_step(args, data, model, device, save_path, compute_metrics):
    """
    테스트 한 step (forward + metric 계산 + 결과 저장)
    """
    number = data['number']
    in_img = data['in_img'].to(device)
    label = data['label'].to(device)
    b, c, h, w = in_img.size()
    w_pad = (math.ceil(w/32)*32 - w) // 2
    h_pad = (math.ceil(h/32)*32 - h) // 2
    w_odd_pad = w_pad
    h_odd_pad = h_pad
    if w % 2 == 1:
        w_odd_pad += 1
    if h % 2 == 1:
        h_odd_pad += 1
    in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
    with torch.no_grad():
        st = time.time()
        out_1, out_2, out_3 = model(in_img)
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


def demo_step(args, data, model, save_path, device):
    """
    - args: config/arguments
    - data: DataLoader에서 받은 {'in_img': tensor, 'number': filename 등}
    - model: 학습된 모델 (eval 모드로 호출)
    - save_path: 결과 저장 경로
    - device: 'cuda' 또는 'cpu'
    """
    model.eval()
    in_img = data['in_img'].to(device)
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

    in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

    with torch.no_grad():
        out_1, out_2, out_3 = model(in_img)
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
