import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import shutil
import time,math
from torch.optim import AdamW
from tensorboardX import SummaryWriter
import torch
from torch.nn.functional import pad
from torchvision import transforms
from PIL import Image
def batch_grid(batch_x, batch_y):
    # batch_x: bsize, samples, bar_size
    bsize, samples, bar_size = batch_x.shape
    res = torch.zeros((bsize, samples, bar_size, bar_size, 2))
    for i in range(bar_size):
        for j in range(bar_size):
            res[:,:,i,j,0] = batch_x[:,:,i]
            res[:,:,i,j,1] = batch_y[:,:,j]
    return res

def make_coords(shape_hr, shape_lr, flatten = True):
    H, W = shape_hr
    h, _ = shape_lr
    scale = H / h
    coord_seqs = []
    seq_h = (1 / scale) * torch.arange(H).float()
    seq_w = (1 / scale) * torch.arange(W).float()
    coord_seqs.append(seq_h)
    coord_seqs.append(seq_w)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img_hr, img_lr, hr_bicubic):
    # img: tensor, (C, H, W)
    channels = img_hr.shape[0]
    coord = make_coords(img_hr.shape[-2 : ], img_lr.shape[-2 : ])
    intensity_hr = img_hr.view(channels, -1).permute(1, 0)
    intensity_bi = hr_bicubic.view(channels, -1).permute(1, 0)
    return coord, intensity_hr, intensity_bi

def make_coords_x(x_lst, degree, mode):
    # x_lst: bsize, sample, bar_size, 1
    if mode == 'poly':
        x_out = torch.ones_like(x_lst)
        for i in range(1, degree + 1):
            x_out = torch.concat([x_out, x_lst**i],axis = 3)
    elif mode == 'fourier':
        x_out = torch.ones_like(x_lst)
        for i in range(1, int((degree) / 2) + 1):
            x_out = torch.concat([x_out, torch.sin(0.2 * i * torch.pi * x_lst)],axis = 3)
            x_out = torch.concat([x_out, torch.cos(0.2 * i * torch.pi * x_lst)],axis = 3)
    return x_out

def make_optimizer(model, optimizer_spec, load_sd=False):
    Optimizer = {
        'adamw': AdamW
    }[optimizer_spec['name']]
    other_params = []
    basis_params = []
    for n, p in model.named_parameters():
        if "basis" in n:
            basis_params.append(p)
        else:
            other_params.append(p)
    if len(basis_params) > 0:
        optimizer = Optimizer(
            [{"params": other_params}, {"params": basis_params, 'weight_decay': 0}],#, 'lr': optimizer_spec['args']['lr']/10}],
            **optimizer_spec['args'])
    else:
        optimizer = Optimizer(other_params, **optimizer_spec['args'])
    # optimizer = Optimizer(model.parameters(), **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

def make_all_coords(num, sr_scale):
    length = num * sr_scale
    gap = 1 / sr_scale
    return torch.tensor([gap * i for i in range(length)], dtype = torch.float32).reshape(1, length)

def pad_tensor(input, bar_size):
    b, S, H, W = input.shape
    if H % bar_size == 0 and W % bar_size == 0:
        return input, (0, 0)
    elif H % bar_size != 0 and W % bar_size == 0:
        pad_size = bar_size - (H % bar_size)
        padded = pad(input, (0,0,0,pad_size), mode = 'constant')
        return padded, (pad_size, 0)
    elif H % bar_size == 0 and W % bar_size != 0:
        pad_size = bar_size - (W % bar_size)
        padded = pad(input, (0,pad_size,0,0), mode = 'constant', value = 0)
        return padded, (0, pad_size)
    else:
        pad_size_w = bar_size - (W % bar_size)
        pad_size_h = bar_size - (H % bar_size)
        padded = pad(input, (0,pad_size_w,0,pad_size_h), mode = 'constant', value = 0)
        return padded, (pad_size_h, pad_size_w)

def pad_tensor_slide(input, pad_size):
    padded = pad(input, (pad_size, pad_size, pad_size, pad_size), mode = 'reflect')
    return padded

def query_intensity(params, coord, bar_size, mode):
    (height, width) = coord
    w = params[height // bar_size][width]
    x = height % bar_size
    return get_voxels(w, x, mode)

def get_voxels(w, x, mode):
    # w: (area, degree + 1)
    if mode == 'poly':
        degree = w.shape[1] - 1
        x_out = torch.ones_like(x)
        for i in range(1, degree + 1):
            x_out = torch.concat([x_out, x**i],axis = 0)
        voxels = torch.matmul(w, x_out)
    elif mode == 'fourier':
        degree = len(w) - 1
        x_out = torch.ones_like(x)
        for i in range(1, int((degree) / 2) + 1):
            x_out = torch.concat([x_out, torch.sin(0.2 * i * torch.pi * x)],axis = -1)
            x_out = torch.concat([x_out, torch.cos(0.2 * i * torch.pi * x)],axis = -1)
        voxels = torch.matmul(x_out, w)
    return voxels

def save_poly(path, w, x_gt, y_gt, bar_size, mode):
    
    fig = plt.figure()
    x_pt = torch.linspace(0, bar_size, 200).reshape(200, 1)
    y_pt = get_voxels(w, x_pt, mode)
    plt.plot(x_pt.cpu().numpy().reshape(200), y_pt.cpu().numpy().reshape(200), color = 'r')
    plt.plot(x_gt, y_gt.reshape(len(x_gt)), color = 'k')
    plt.plot(x_gt, y_gt.reshape(len(x_gt)), 'o', color = 'k')
    fig.savefig(path)
    plt.close()

def save_poly_test(path, w, x_gt, y_gt, y_pt_points, bar_size, mode):
    
    fig = plt.figure()
    x_pt = torch.linspace(0, bar_size, 17).reshape(17, 1)
    y_pt = get_voxels(w, x_pt, mode)
    plt.plot(x_pt.cpu().numpy().reshape(17), y_pt.cpu().numpy().reshape(17), color = 'r')
    plt.plot(x_gt, y_pt_points.reshape(len(x_gt)), color = 'b')
    plt.plot(x_gt, y_pt_points.reshape(len(x_gt)), 'o', color = 'b')
    plt.plot(x_gt, y_gt.reshape(len(x_gt)), color = 'k')
    plt.plot(x_gt, y_gt.reshape(len(x_gt)), 'o', color = 'k')
    fig.savefig(path)
    plt.close()
    return y_pt


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    # ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer

def set_test_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    # writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log

class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v
    
def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.3f}M'.format(tot / 1e6)
        else:
            return '{:.3f}K'.format(tot / 1e3)
    else:
        return tot
    
def calc_psnr(sr, hr, rgb_range=1):
    diff = (sr - hr) / rgb_range
    mse = diff.pow(2).mean()
    return -10 * torch.log10(mse + 1e-10)

def resize_img_to_out_size(img, out_size):
    h, w = img.shape[-2:]
    if h < out_size and w < out_size:
        h_new, w_new = out_size, out_size
        img_new = transforms.ToTensor()(transforms.Resize((h_new, w_new), Image.Resampling.BICUBIC)(transforms.ToPILImage()(img)))
    elif h < out_size and w >= out_size:
        h_new, w_new = out_size, w
        img_new = transforms.ToTensor()(transforms.Resize((h_new, w_new), Image.Resampling.BICUBIC)(transforms.ToPILImage()(img)))
    elif h >= out_size and w < out_size:
        h_new, w_new = h, out_size
        img_new = transforms.ToTensor()(transforms.Resize((h_new, w_new), Image.Resampling.BICUBIC)(transforms.ToPILImage()(img)))
    else:
        img_new = img
    return img_new

def crop_center(img, h_new, w_new):
    h, w = img.shape[-2:]
    img_new = img[:,(h//2 - h_new//2):(h//2 + h_new//2), (w//2 - w_new//2):(w//2 + w_new//2)]
    return img_new


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1) / width
    pos_h = torch.arange(0., height).unsqueeze(1) / height
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe