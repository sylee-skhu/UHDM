import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from datetime import datetime
from tqdm import tqdm
from models import create_model
from datasets import create_dataset
from utils.common import *
from config.config import get_parser
import logging


def test_step(args, data, model, device, save_path, compute_metrics):
    number = data['number']
    label = data['label']

    data_mod, h_pad, h_odd_pad, w_pad, w_odd_pad = pad_and_replace(data)

    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out_1 = model(data_mod)[0]
        end.record()
        torch.cuda.synchronize()
        cur_time = start.elapsed_time(end) / 1000.0  # ms → sec

        # crop padding
        if h_pad != 0:
            out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
        if w_pad != 0:
            out_1 = out_1[:, :, :, w_pad:-w_odd_pad]

    # metric 계산
    cur_lpips, cur_psnr, cur_ssim = 0.0, 0.0, 0.0
    if args.EVALUATION_METRIC and compute_metrics is not None:
        cur_lpips, cur_psnr, cur_ssim = compute_metrics.compute(out_1, label)

    # save images (마스터만 저장)
    if args.SAVE_IMG and dist.get_rank() == 0:
        out_save = out_1.detach().cpu()
        torchvision.utils.save_image(
            out_save,
            os.path.join(save_path, f'test_{number[0]}.{args.SAVE_IMG}')
        )
    return cur_psnr, cur_ssim, cur_lpips, cur_time


def test(args, TestImgLoader, model, device, save_path, compute_metrics):
    tbar = tqdm(TestImgLoader) if dist.get_rank() == 0 else TestImgLoader
    total_psnr, total_ssim, total_lpips, total_time = 0, 0, 0, 0
    for batch_idx, data in enumerate(tbar):
        data = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in data.items()}
        model.eval()
        cur_psnr, cur_ssim, cur_lpips, cur_time = test_step(args, data, model, device, save_path, compute_metrics)
        number = data['number']
        total_psnr += cur_psnr
        total_ssim += cur_ssim
        total_lpips += cur_lpips
        if batch_idx > 5:  # warm-up 패스
            total_time += cur_time
            avg_time = total_time / (batch_idx - 5)
        else:
            avg_time = 0.0
        avg_psnr = total_psnr / (batch_idx + 1)
        avg_ssim = total_ssim / (batch_idx + 1)
        avg_lpips = total_lpips / (batch_idx + 1)

        if dist.get_rank() == 0:
            if args.EVALUATION_METRIC:
                desc = f"Test: LPIPS={avg_lpips:.4f}, PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}"
            elif args.EVALUATION_TIME:
                desc = f"Avg. TIME={avg_time:.4f}s"
            else:
                desc = "Test: No evaluation"
            tbar.set_description(desc)
            tbar.update()
            if args.EVALUATION_METRIC:
                logging.info('%s: LPIPS %.4f, PSNR %.4f, SSIM %.4f', number[0], cur_lpips, cur_psnr, cur_ssim)
            if args.EVALUATION_TIME:
                logging.info('%s: TIME %.4f s', number[0], cur_time)

    if dist.get_rank() == 0:
        if args.EVALUATION_METRIC:
            logging.warning('Avg. LPIPS=%.4f, PSNR=%.4f, SSIM=%.4f', avg_lpips, avg_psnr, avg_ssim)
        if args.EVALUATION_TIME:
            logging.warning('Avg. TIME=%.4f s', avg_time)


def load_checkpoint(args, model):
    if args.LOAD_PATH:
        load_path = args.LOAD_PATH
        save_path = os.path.join(args.TEST_RESULT_DIR, 'customer')
        log_path = os.path.join(args.TEST_RESULT_DIR, 'customer_result.log')
    else:
        load_epoch = args.TEST_EPOCH
        if load_epoch == 'auto':
            load_path = os.path.join(args.NETS_DIR, 'checkpoint_latest.tar')
            save_path = os.path.join(args.TEST_RESULT_DIR, 'latest')
            log_path = os.path.join(args.TEST_RESULT_DIR, 'latest_result.log')
        else:
            load_path = os.path.join(args.NETS_DIR, f'checkpoint_{load_epoch:06d}.tar')
            save_path = os.path.join(args.TEST_RESULT_DIR, f'{load_epoch:04d}')
            log_path = os.path.join(args.TEST_RESULT_DIR, f'{load_epoch:04d}_result.log')
    if dist.get_rank() == 0:
        mkdir(save_path)
    if load_path.endswith('.pth'):
        model_state_dict = torch.load(load_path)
    else:
        model_state_dict = torch.load(load_path)['state_dict']
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    return load_path, save_path, log_path


def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def main():
    args = get_parser()
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dist.init_process_group('nccl')

    args.TEST_RESULT_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'test_result')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    if rank == 0:
        mkdir(args.TEST_RESULT_DIR)

    set_seed(args.SEED)
    # DataLoader
    test_path = args.TEST_DATASET
    args.BATCH_SIZE = 1
    TestImgLoader = create_dataset(args, data_path=test_path, mode='test', device=device)

    # Model
    model = create_model(args).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    load_path, save_path, log_path = load_checkpoint(args, model)

    set_logging(log_path)
    logging.warning(datetime.now())
    logging.warning('load model from %s', load_path)
    logging.warning('save image results to %s', save_path)
    logging.warning('save logger to %s', log_path)

    # Metric
    compute_metrics = None
    if args.EVALUATION_TIME:
        args.EVALUATION_METRIC = False
    if args.EVALUATION_METRIC:
        from utils.metric import create_metrics
        compute_metrics = create_metrics(args, device=device)

    # Test
    test(args, TestImgLoader, model, device, save_path, compute_metrics)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
