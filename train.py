import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import torchvision
from models import create_model
from datasets import create_dataset
from losses import create_loss
from tqdm import tqdm
from utils.common import *
import numpy as np
import random
from config.config import get_parser


def train_step(args, data, model, loss_fn, device, iters):
    model.train()
    outputs = model(data)
    in_img = data['in_img']
    label = data['label']
    loss = loss_fn(outputs, label)
    if iters % args.SAVE_ITER == (args.SAVE_ITER - 1) and dist.get_rank() == 0:  # master만 저장
        in_save = in_img.detach().cpu()
        out_save = outputs[0].detach().cpu()
        gt_save = label.detach().cpu()
        res_save = torch.cat((in_save, out_save, gt_save), 3)
        save_number = (iters + 1) // args.SAVE_ITER
        torchvision.utils.save_image(
            res_save,
            args.VISUALS_DIR + '/visual_x%04d_' % args.SAVE_ITER + '%05d' % save_number + '.jpg'
        )
    return loss


def train_epoch(args, TrainImgLoader, model, loss_fn, optimizer, device, epoch, iters, lr_scheduler):
    model.train()
    tbar = tqdm(TrainImgLoader) if dist.get_rank() == 0 else TrainImgLoader
    total_loss = 0
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    for batch_idx, data in enumerate(tbar):
        loss = train_step(args, data, model, loss_fn, device, iters)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iters += 1
        total_loss += loss.item()
        avg_train_loss = total_loss / (batch_idx + 1)
        if dist.get_rank() == 0:
            desc = 'Training  : Epoch %d, lr %.7f, Avg. Loss = %.5f' % (epoch, lr, avg_train_loss)
            tbar.set_description(desc)
            tbar.update()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    lr_scheduler.step()
    return lr, avg_train_loss, iters


def load_checkpoint(model, optimizer, load_epoch):
    load_dir = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
    print('Loading pre-trained checkpoint %s' % load_dir)
    ckpt = torch.load(load_dir, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    learning_rate = ckpt['learning_rate']
    iters = ckpt['iters']
    print('Learning rate recorded from the checkpoint: %s' % str(learning_rate))
    return learning_rate, iters


def main():
    args = get_parser()
    # torchrun이 환경변수로 넘겨줌
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    # DDP 초기화
    dist.init_process_group('nccl')

    # 디렉토리 등 각 프로세스가 만들어도 안전
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'logs')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    args.VISUALS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'train_visual')
    if rank == 0:
        mkdir(args.LOGS_DIR)
        mkdir(args.NETS_DIR)
        mkdir(args.VISUALS_DIR)
    logger = SummaryWriter(args.LOGS_DIR) if rank == 0 else None

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

    model = create_model(args).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.BASE_LR}], betas=(0.9, 0.999))
    learning_rate = args.BASE_LR
    iters = 0
    if args.LOAD_EPOCH:
        learning_rate, iters = load_checkpoint(model, optimizer, args.LOAD_EPOCH)

    loss_fn = create_loss(args).to(device)

    train_path = args.TRAIN_DATASET

    sampler = DistributedSampler(TrainDataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    TrainImgLoader = create_dataset(args, data_path=train_path, mode='train', device=device, sampler=sampler)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_MULT, eta_min=args.ETA_MIN,
        last_epoch=args.LOAD_EPOCH - 1
    )

    print(f"**** Rank {rank} (local_rank {args.local_rank}) start training! ****")
    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        sampler.set_epoch(epoch)
        learning_rate, avg_train_loss, iters = train_epoch(
            args, TrainImgLoader, model, loss_fn, optimizer, device, epoch, iters, lr_scheduler
        )
        if rank == 0 and logger:
            logger.add_scalar('Train/avg_loss', avg_train_loss, epoch)
            logger.add_scalar('Train/learning_rate', learning_rate, epoch)
        if rank == 0:
            # Save the network per ten epoch
            if epoch % 10 == 0:
                savefilename = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % epoch + '.tar'
                torch.save({
                    'learning_rate': learning_rate,
                    'iters': iters,
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.module.state_dict()
                }, savefilename)
            savefilename = args.NETS_DIR + '/checkpoint' + '_' + 'latest.tar'
            torch.save({
                'learning_rate': learning_rate,
                'iters': iters,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.module.state_dict()
            }, savefilename)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
