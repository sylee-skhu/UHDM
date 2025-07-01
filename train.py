import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
import torchvision
from models import create_model
from datasets import create_dataset
from losses import create_loss
from tqdm import tqdm
from utils.common import *
import numpy as np
from config.config import get_parser


def train_step(args, data, model, loss_fn, optimizer_g, optimizer_d, device, iters):
    model.train()
    outputs = model(data)
    in_img = data['in_img']
    label = data['label']

    # ==== Generator update ====
    loss_g = loss_fn(outputs, label)
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    loss_d = 0.0
    if args.USE_GAN:
        for d_iter in range(getattr(args, 'D_ITERS', 1)):
            # ==== Discriminator update ====
            outputs_d = model(data)
            loss_d_cur = loss_fn(outputs_d, label, d_loss=True)
            optimizer_d.zero_grad()
            loss_d_cur.backward()
            optimizer_d.step()
            loss_d += loss_d_cur.item()
        loss_d /= getattr(args, 'D_ITERS', 1)  # 평균 (optional)

    # Visualization (G output만 저장)
    if iters % args.SAVE_ITER == (args.SAVE_ITER - 1) and dist.get_rank() == 0:
        in_save = in_img.detach().cpu()
        out_save = outputs[0].detach().cpu()
        gt_save = label.detach().cpu()
        res_save = torch.cat((in_save, out_save, gt_save), 3)
        save_number = (iters + 1) // args.SAVE_ITER
        torchvision.utils.save_image(
            res_save,
            args.VISUALS_DIR + '/visual_x%04d_' % args.SAVE_ITER + '%05d' % save_number + '.jpg'
        )
    return loss_g.item(), loss_d


def train_epoch(args, TrainImgLoader, model, loss_fn, optimizer_g, optimizer_d, device, epoch, iters, lr_scheduler):
    model.train()
    tbar = tqdm(TrainImgLoader) if dist.get_rank() == 0 else TrainImgLoader
    total_loss_g = 0
    total_loss_d = 0
    lr = optimizer_g.state_dict()['param_groups'][0]['lr']
    for batch_idx, data in enumerate(tbar):
        data = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in data.items()}
        loss_g, loss_d = train_step(args, data, model, loss_fn, optimizer_g, optimizer_d, device, iters)
        iters += 1
        total_loss_g += loss_g
        total_loss_d += loss_d
        avg_train_loss_g = total_loss_g / (batch_idx + 1)
        avg_train_loss_d = total_loss_d / (batch_idx + 1)
        if dist.get_rank() == 0:
            desc = f"Epoch {epoch}, lr {lr:.7f}, G Loss={avg_train_loss_g:.5f}, D Loss={avg_train_loss_d:.5f}"
            tbar.set_description(desc)
            tbar.update()
    lr_scheduler.step()
    return lr, avg_train_loss_g, avg_train_loss_d, iters


def load_checkpoint(args, model, optimizer_g, optimizer_d):
    load_path = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % args.LOAD_EPOCH + '.tar'
    print('Loading pre-trained checkpoint %s' % load_path)
    ckpt = torch.load(load_path, map_location='cpu')

    model_state_dict = ckpt['state_dict']
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    optimizer_g.load_state_dict(ckpt['optimizer_g'])
    if optimizer_d is not None and 'optimizer_d' in ckpt:
        optimizer_d.load_state_dict(ckpt['optimizer_d'])

    learning_rate = ckpt['learning_rate']
    iters = ckpt['iters']
    print('Learning rate recorded from the checkpoint: %s' % str(learning_rate))
    return learning_rate, iters


def main():
    args = get_parser()
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dist.init_process_group('nccl')

    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'logs')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    args.VISUALS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'train_visual')
    if rank == 0:
        mkdir(args.LOGS_DIR)
        mkdir(args.NETS_DIR)
        mkdir(args.VISUALS_DIR)
    logger = SummaryWriter(args.LOGS_DIR) if rank == 0 else None

    set_seed(args.SEED)

    model = create_model(args).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.USE_GAN:
        optimizer_g = optim.Adam([{'params': model.module.G.parameters(), 'initial_lr': args.BASE_LR}], betas=(0.9, 0.999))
        optimizer_d = optim.Adam([{'params': model.module.D.parameters(), 'initial_lr': args.BASE_LR}], betas=(0.9, 0.999))
    else:
        optimizer_g = optim.Adam([{'params': model.module.parameters(), 'initial_lr': args.BASE_LR}], betas=(0.9, 0.999))
        optimizer_d = None

    learning_rate = args.BASE_LR
    iters = 0
    if args.LOAD_EPOCH:
        learning_rate, iters = load_checkpoint(args, model, optimizer_g, optimizer_d)

    loss_fn = create_loss(args).to(device)

    TrainImgLoader = create_dataset(args, data_path=args.TRAIN_DATASET, mode='train', device=device)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_g, T_0=args.T_0, T_mult=args.T_MULT, eta_min=args.ETA_MIN,
        last_epoch=args.LOAD_EPOCH - 1
    )

    print(f"**** Rank {rank} (local_rank {args.local_rank}) start training! ****")
    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        if hasattr(TrainImgLoader, 'sampler') and TrainImgLoader.sampler is not None:
            TrainImgLoader.sampler.set_epoch(epoch)
        lr, avg_loss_g, avg_loss_d, iters = train_epoch(
            args, TrainImgLoader, model, loss_fn, optimizer_g, optimizer_d, device, epoch, iters, lr_scheduler
        )
        if rank == 0 and logger:
            logger.add_scalar('Train/avg_loss_g', avg_loss_g, epoch)
            logger.add_scalar('Train/avg_loss_d', avg_loss_d, epoch)
            logger.add_scalar('Train/learning_rate', lr, epoch)
        if rank == 0:
            # Save the network per ten epoch
            ckpt = {
                'learning_rate': lr,
                'iters': iters,
                'optimizer_g': optimizer_g.state_dict(),
                'state_dict': model.module.state_dict()
            }
            if args.USE_GAN:
                ckpt['optimizer_d'] = optimizer_d.state_dict()
            if epoch % 10 == 0:
                savefilename = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % epoch + '.tar'
                torch.save(ckpt, savefilename)
            savefilename = args.NETS_DIR + '/checkpoint' + '_' + 'latest.tar'
            torch.save(ckpt, savefilename)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
