import os, sys
import torch
from torch.utils.data.dataloader import DataLoader
import argparse
import utils
import torch.nn as nn
import models
import warnings
from tqdm import tqdm
import yaml,math
import warnings
import datasets
import random
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
warnings.filterwarnings("ignore")
from torchvision import transforms

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def make_data_loader(spec, world_size, rank, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    data_sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=(tag=='train'))
    if tag == 'train':
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
            shuffle=False, num_workers=8, pin_memory=True, sampler=data_sampler)
    else:
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
            shuffle=False, num_workers=8, pin_memory=True)
    return loader


def make_data_loaders(world_size, rank):
    train_loader = make_data_loader(config.get('train_dataset'), world_size, rank, tag='train')
    val_loader_x2 = make_data_loader(config.get('val_dataset_x2'), world_size, rank, tag='val_x2')
    # val_loader_x3 = make_data_loader(config.get('val_dataset_x3'), tag='val_x3')
    val_loader_x4 = make_data_loader(config.get('val_dataset_x4'), world_size, rank, tag='val_x4')
    return train_loader, val_loader_x2, val_loader_x4

def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model'])
        if config.get('edsr_pretrain') is not None:
            edsr_ckpt = torch.load(config['edsr_pretrain'])
            model_dict = model.state_dict()
            pretrained_dict = {'encoder.' + k: v for k, v in edsr_ckpt.items() if 'encoder.' + k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            log('edsr pretrain model loaded!')
        if config.get('checkpoint') is not None:
            checkpoint = torch.load(config['checkpoint'])
            model.load_state_dict(checkpoint, strict=True)
            log('checkpoint loaded!')
        # optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        optimizer = utils.make_optimizer(model, config['optimizer'])
        epoch_start = config['epoch_start']
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
    if int(os.environ['LOCAL_RANK']) == 0:
        log('{} model: #params={}'.format(os.environ['LOCAL_RANK'], utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model, optimizer, scheduler, epoch=0):
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()
    num_iter = len(train_loader)
    for i, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v[0].to(rank)
        gt = batch['gt']
        pred = model(batch['inp'], batch['coord'], batch['cell'])
        pred  = pred + batch['hr_res']
        loss = loss_fn(pred, gt)
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        #######################################
        # by kai: set learning rate
        # lr = scheduler.step()
        # for pg in optimizer.param_groups:
        #     pg['lr'] = lr
        #######################################
        lr = scheduler.get_last_lr()[0]
        optimizer.step()
        if (i+1) % 100 == 0: #or i == 0 or (i+1) == len(train_loader):
            train_loss_cp = torch.tensor(train_loss.item()).to(rank)
            dist.all_reduce(train_loss_cp)
            avg_train_loss = train_loss_cp.item() / world_size
            if rank == 0:                
                log_info = [f'iter {i+1}/{num_iter} lr={lr:.3e} ']
                log_info.append('train loss = {:.6f}'.format(avg_train_loss))
                log(', '.join(log_info))
                writer.add_scalar('lr', lr, i+(epoch-1)*num_iter)

        pred = None; loss = None; train_loss_cp = None

    train_loss_cp = torch.tensor(train_loss.item()).to(rank)
    dist.all_reduce(train_loss_cp)
    avg_train_loss = train_loss_cp.item() / world_size
    return avg_train_loss

def val(val_loader, model,scale,epoch,save_path):
    rank = int(os.environ['LOCAL_RANK'])
    if rank == 0:
        save_img_dir = os.path.join(save_path,'val_img')
        if not os.path.exists(save_img_dir): os.mkdir(save_img_dir)
    model.eval()
    model.module.cal_x_y_coord()
    loss_fn = nn.L1Loss()
    val_loss = utils.Averager()
    val_psnr = utils.Averager()
    if rank == 0:
        cnt = 0
        for batch in tqdm(val_loader, leave=False, desc='validating:x'+str(scale)+'...'):
            for k, v in batch.items():
                batch[k] = v.to(rank)
            pred = model(batch['inp'], batch['coord'], batch['cell'])
            pred = pred + batch['hr_res']
            cnt+=1
            if cnt==7:
                img_path = os.path.join(save_img_dir, 'val_e'+str(epoch)+'_x'+str(scale)+'.png')
                transforms.ToPILImage()(pred.clamp(0,1).reshape(1,int(math.sqrt(pred.shape[1])),int(math.sqrt(pred.shape[1])))).save(img_path)
            gt = batch['gt']
            loss = loss_fn(pred, gt)
            psnr = utils.calc_psnr(pred, gt)
            val_loss.add(loss.item())
            val_psnr.add(psnr.item())
            pred = None; loss = None
    else:
        for batch in val_loader:
            for k, v in batch.items():
                batch[k] = v.to(rank)
            pred = model(batch['inp'], batch['coord'], batch['cell'])
            pred = pred + batch['hr_res']
            gt = batch['gt']
            loss = loss_fn(pred, gt)
            psnr = utils.calc_psnr(pred, gt)
            val_loss.add(loss.item())
            val_psnr.add(psnr.item())
            pred = None; loss = None

    return val_loss.item(), val_psnr.item()

def main(args):
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl',init_method='env://',world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # print(rank, world_size)

    max_val_v = -1e18
    global config, log, writer
    config = args.config
    save_path = args.save_path
    if rank == 0:
        log, writer = utils.set_save_path(save_path)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)
            checkpoint_dir = os.path.join(save_path, 'checkpoints')
        if config['save_checkpoint'] and not(os.path.exists(checkpoint_dir)): os.makedirs(checkpoint_dir)

    train_loader, val_loader_x2, val_loader_x4 = make_data_loaders(world_size, rank)
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model.to(rank),device_ids=[torch.cuda.current_device()])
    #################################################################
    # by Kai
    # lr_scheduler = CosineScheduler(
    #     max_iters=len(train_loader)*config['epoch'],
    #     warmup_iters=len(train_loader),
    #     warmup_init_lr=1e-6,
    #     max_lr=1e-4,
    #     min_lr=1e-6)
    #################################################################

    epoch_max = config['epoch']
    if rank == 0:timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        if rank == 0:t_epoch_start = timer.t()
        if rank == 0:log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        train_loss = train(train_loader, model, optimizer, lr_scheduler, epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if rank == 0:
            log_info.append('train loss = {:.6f}'.format(train_loss))
            writer.add_scalars('loss', {'loss': train_loss}, epoch)
        
        _, val_psnr_x2 = val(val_loader_x2, model,2,epoch,save_path)
        _, val_psnr_x4 = val(val_loader_x4, model,4,epoch,save_path)
        if rank == 0:
            log_info.append('val_x2: psnr = {:.5f}'.format(val_psnr_x2))
            log_info.append('val_x4: psnr = {:.5f}'.format(val_psnr_x4))
            writer.add_scalars('psnr_x2', {'val_x2': val_psnr_x2}, epoch)
            writer.add_scalars('psnr_x4', {'val_x4': val_psnr_x4}, epoch)
            t = timer.t()
            prog = (epoch) / (epoch_max)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            model_ = model
            if epoch >= config['epoch_start_save'] and (epoch - config['epoch_start_save']) % config['epoch_every_save'] == 0:
                torch.save(model_.state_dict(), os.path.join(checkpoint_dir, 'checkpoint_' + str(epoch) + '.pt'))
            if val_psnr_x2 + val_psnr_x4 > max_val_v:
                max_val_v = val_psnr_x2 + val_psnr_x4
                torch.save(model_.state_dict(), os.path.join(save_path, 'epoch-best.pt'))

            log(', '.join(log_info))
            writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--local_rank')
    parser.add_argument('--dir')
    parser.add_argument('--name')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args.config = yaml.load(f, Loader = yaml.FullLoader)
        if int(os.environ['LOCAL_RANK']) == 0:
            print('config loaded.')

    name = args.name
    args.save_path = os.path.join('.', args.dir, name)
    main(args)

