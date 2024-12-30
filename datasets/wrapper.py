import random
import numpy as np
import torch, scipy
from torch.utils.data import Dataset
from torchvision import transforms
from utils import *
from datasets import register
from utils import to_pixel_samples, crop_center
from PIL import Image
import sys
# import albumentations as A
import math
def resize_image_row(img, h_lr):
    # if not hasattr(resize_image_row, 'printed'):
    #     print('Downsampled method: Row choosing')
    #     resize_image_row.printed = True
    _, h, w = img.shape
    rows = torch.arange(0, h, h // h_lr)
    img_tensor = img[:,rows, :]
    return img_tensor

def resize_image_bicubic(img, h,w):
    # if not hasattr(resize_image_bicubic, 'printed'):
    #     # print('Downsampled method: bicubic')
    #     resize_image_bicubic.printed = True
    img_tensor = transforms.ToTensor()(transforms.Resize((h,w), Image.Resampling.BICUBIC)(transforms.ToPILImage()(img)))
    return img_tensor

def up_SP_downsample(hr, h_lr, h_hr, hr_d):
    lr_d = hr_d * (h_hr / h_lr)
    _, _, W = hr.shape 
    # lr_slices = round(H / scale)
    hr_coords = torch.tensor([(t * hr_d + hr_d / 2) for t in range(h_hr)])
    tensor_norm = torch.tensor(np.real(scipy.io.loadmat('../breast_sl_profile.mat')['sl_profile']))
    lr = torch.ones((h_lr, W))
    for i in range(h_lr):
        win_s, win_e = i * lr_d, (i + 1) * lr_d
        selected_indices = torch.where((hr_coords >= win_s) & (hr_coords <= win_e))
        selected_slices = hr[0][selected_indices]
        selected_slices_up = transforms.ToTensor()(transforms.Resize((1000, W), Image.Resampling.BILINEAR)(transforms.ToPILImage()(selected_slices)))
        lr[i] = tensor_norm.T.float() @ selected_slices_up
    lr = lr.unsqueeze(0)
    return lr

def kspace_zero_padding_resample_torch(input, h_new, w_new, resize=True):
    h, w = input.shape[-2:]
    f = torch.fft.fftshift(torch.fft.fft2(input))
    dh, dw = ((h_new - h) // 2, (w_new - w) // 2)
    if dh >= 0 and dw >= 0:
        f_resized = torch.zeros((1,h_new, w_new), dtype=torch.complex64, device=input.device)
        f_resized[:,dh:dh+h, dw:dw+w] = f
        img_ds = torch.fft.ifft2(torch.fft.ifftshift(f_resized))
        img_ds = torch.real(img_ds).clamp(0,1)
    else:
        f_resized = torch.zeros((1,h, w), dtype=torch.complex64, device=input.device)
        if h_new % 2 != 0:
            f_resized[:,(h//2 - h_new//2 - 1):(h//2 + h_new//2), 
                    (w//2 - w_new//2 - 1):(w//2 + w_new//2)] = f[:,(h//2 - h_new//2 - 1):(h//2 + h_new//2), 
                                                            (w//2 - w_new//2 - 1):(w//2 + w_new//2)]
        else:
            f_resized[:,(h//2 - h_new//2):(h//2 + h_new//2), 
                    (w//2 - w_new//2):(w//2 + w_new//2)] = f[:,(h//2 - h_new//2):(h//2 + h_new//2), 
                                                            (w//2 - w_new//2):(w//2 + w_new//2)]            
        img_ds = torch.fft.ifft2(torch.fft.ifftshift(f_resized))
        img_ds = torch.real(img_ds).clamp(0,1) 
        if resize:
            img_ds = transforms.ToTensor()(transforms.Resize((h_new, w_new), 
                                                         Image.Resampling.BICUBIC)(transforms.ToPILImage()(img_ds)))      
    return img_ds

@register('sr-explicit-downsampled-kspace')
class MRIWrapper(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        if self.inp_size is None:
            h_hr, w_hr = img.shape[-2], img.shape[-1]
            if h_hr > w_hr:
                h_hr = w_hr
                img = img[:, :h_hr, :w_hr]
            elif h_hr < w_hr:
                w_hr = h_hr
                img = img[:, :h_hr, :w_hr]
            h_lr = math.floor(h_hr / s + 1e-9)
            w_lr = math.floor(w_hr / s + 1e-9)
            while w_lr % 4 != 0: w_lr  = w_lr - 1
            while h_lr % 4 != 0: h_lr  = h_lr - 1
            img_down = kspace_zero_padding_resample_torch(img, h_lr, w_lr) #resize_image_bicubic(img, h_lr, w_lr)
            crop_lr, crop_hr = img_down, img
            crop_res = resize_image_bicubic(crop_lr, h_hr, w_hr) #kspace_zero_padding_resample_torch(crop_lr, h_hr, w_hr) #
            cell = torch.tensor([h_lr/h_hr, w_lr/w_hr])
        else:
            H_hr, W_hr = img.shape[-2], img.shape[-1]
            H_lr, W_lr = math.floor(H_hr / s), math.floor(W_hr / s)
            img_lr = kspace_zero_padding_resample_torch(img, H_lr, W_lr)
            img_res_hr = resize_image_bicubic(img_lr, H_hr, W_hr) #kspace_zero_padding_resample_torch(img_lr, H_hr, W_hr)
            h_lr, w_lr = self.inp_size, self.inp_size
            h_hr, w_hr = round(h_lr * s), round(w_lr * s)
            cell = torch.tensor([h_lr/h_hr, w_lr/w_hr])
            x_lr, y_lr = random.randint(0, H_lr - h_lr - 1), random.randint(0, W_lr - w_lr - 1)
            x_hr, y_hr = math.floor(x_lr * s), math.floor(y_lr * s)
            crop_hr = img[:, x_hr : x_hr + h_hr, y_hr : y_hr + w_hr]
            crop_res = img_res_hr[:, x_hr : x_hr + h_hr, y_hr : y_hr + w_hr]
            crop_lr = img_lr[:, x_lr : x_lr + h_lr, y_lr : y_lr + w_lr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5
            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            crop_res = augment(crop_res)

        hr_coord, hr_gt, hr_res = to_pixel_samples(crop_hr.contiguous(), crop_lr, crop_res.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_gt = hr_gt[sample_lst]
            hr_res = hr_res[sample_lst]
        
        return {
            'inp': crop_lr.float(),
            'coord': hr_coord,
            'hr_res': hr_res,
            'gt': hr_gt.float(),
            'cell': cell
        }





@register('sr-explicit-downsampled-new')
class MRINewWrapper(Dataset):

    def __init__(self, dataset, mode, repeat=None, out_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.out_size = out_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.repeat = repeat
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.mode == 'test':
            s = random.uniform(self.scale_min, self.scale_max)
            img_hr = self.dataset[idx]
            h_hr, w_hr = self.out_size, self.out_size
            h_lr, w_lr = math.floor(h_hr / s + 1e-9), math.floor(w_hr / s + 1e-9)
            crop_hr = crop_center(img_hr, h_hr, w_hr)
            crop_lr = kspace_zero_padding_resample_torch(crop_hr, h_lr, w_lr)
            crop_res_hr = resize_image_bicubic(crop_lr, h_hr, w_hr)
            hr_coord, hr_gt, hr_res = to_pixel_samples(crop_hr.contiguous(), crop_lr, crop_res_hr.contiguous())
            inp_stack = crop_lr
            coord_stack = hr_coord
            hr_res_stack = hr_res
            gt_stack = hr_gt
            cell_stack = torch.tensor([h_lr/h_hr, w_lr/w_hr])

        if self.mode == 'train':
            s = random.uniform(self.scale_min, self.scale_max)
            img_hr = self.dataset[idx]
            H, W = img_hr.shape[-2:]
            h_hr, w_hr = self.out_size, self.out_size
            h_lr, w_lr = math.floor(h_hr / s + 1e-9), math.floor(w_hr / s + 1e-9)
            cell = torch.tensor([h_lr/h_hr, w_lr/w_hr])
            inp_stack = torch.zeros((self.repeat, 1, h_lr, w_lr))
            if self.sample_q is None: num_pixels = h_hr * w_hr
            else: num_pixels = self.sample_q
            coord_stack = torch.zeros((self.repeat, num_pixels, 2))
            hr_res_stack = torch.zeros((self.repeat, num_pixels, 1))
            gt_stack = torch.zeros((self.repeat, num_pixels, 1))
            cell_stack = cell.unsqueeze(0).repeat(self.repeat, 1)
            for i in range(self.repeat):
                x0, y0 = random.randint(0, H - h_hr), random.randint(0, W - w_hr)
                crop_hr = img_hr[:, x0: x0 + w_hr, y0: y0 + h_hr]
                crop_lr = kspace_zero_padding_resample_torch(crop_hr, h_lr, w_lr)
                crop_res_hr = resize_image_bicubic(crop_lr, h_hr, w_hr)

                if self.augment:
                    hflip = random.random() < 0.5
                    vflip = random.random() < 0.5
                    dflip = random.random() < 0.5
                    def augment(x):
                        if hflip:
                            x = x.flip(-2)
                        if vflip:
                            x = x.flip(-1)
                        if dflip:
                            x = x.transpose(-2, -1)
                        return x
                    crop_lr = augment(crop_lr)
                    crop_hr = augment(crop_hr)
                    crop_res_hr = augment(crop_res_hr)
                hr_coord, hr_gt, hr_res = to_pixel_samples(crop_hr.contiguous(), crop_lr, crop_res_hr.contiguous())
                if self.sample_q is not None:
                    sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
                    hr_coord = hr_coord[sample_lst]
                    hr_gt = hr_gt[sample_lst]
                    hr_res = hr_res[sample_lst]
                inp_stack[i] = crop_lr
                coord_stack[i] = hr_coord
                hr_res_stack[i] = hr_res
                gt_stack[i] = hr_gt


        return {
            'inp': inp_stack.float(),
            'coord': coord_stack,
            'hr_res': hr_res_stack,
            'gt': gt_stack.float(),
            'cell': cell_stack
        }

@register('sr-explicit-downsampled-ct')
class MRINewWrapper(Dataset):

    def __init__(self, dataset, mode, repeat=None, out_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.out_size = out_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.repeat = repeat
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.mode == 'test':
            s = random.uniform(self.scale_min, self.scale_max)
            img_hr = self.dataset[idx]
            h_hr, w_hr = self.out_size, self.out_size
            h_lr, w_lr = math.floor(h_hr / s + 1e-9), math.floor(w_hr / s + 1e-9)
            crop_hr = crop_center(img_hr, h_hr, w_hr)
            crop_lr = resize_image_bicubic(crop_hr, h_lr, w_lr)
            crop_res_hr = resize_image_bicubic(crop_lr, h_hr, w_hr)
            hr_coord, hr_gt, hr_res = to_pixel_samples(crop_hr.contiguous(), crop_lr, crop_res_hr.contiguous())
            inp_stack = crop_lr
            coord_stack = hr_coord
            hr_res_stack = hr_res
            gt_stack = hr_gt
            cell_stack = torch.tensor([h_lr/h_hr, w_lr/w_hr])

        if self.mode == 'train':
            s = random.uniform(self.scale_min, self.scale_max)
            img_hr = self.dataset[idx]
            H, W = img_hr.shape[-2:]
            h_hr, w_hr = self.out_size, self.out_size
            h_lr, w_lr = math.floor(h_hr / s + 1e-9), math.floor(w_hr / s + 1e-9)
            cell = torch.tensor([h_lr/h_hr, w_lr/w_hr])
            inp_stack = torch.zeros((self.repeat, 1, h_lr, w_lr))
            if self.sample_q is None: num_pixels = h_hr * w_hr
            else: num_pixels = self.sample_q
            coord_stack = torch.zeros((self.repeat, num_pixels, 2))
            hr_res_stack = torch.zeros((self.repeat, num_pixels, 1))
            gt_stack = torch.zeros((self.repeat, num_pixels, 1))
            cell_stack = cell.unsqueeze(0).repeat(self.repeat, 1)
            for i in range(self.repeat):
                x0, y0 = random.randint(0, H - h_hr), random.randint(0, W - w_hr)
                crop_hr = img_hr[:, x0: x0 + w_hr, y0: y0 + h_hr]
                crop_lr = resize_image_bicubic(crop_hr, h_lr, w_lr)
                crop_res_hr = resize_image_bicubic(crop_lr, h_hr, w_hr)

                if self.augment:
                    hflip = random.random() < 0.5
                    vflip = random.random() < 0.5
                    dflip = random.random() < 0.5
                    def augment(x):
                        if hflip:
                            x = x.flip(-2)
                        if vflip:
                            x = x.flip(-1)
                        if dflip:
                            x = x.transpose(-2, -1)
                        return x
                    crop_lr = augment(crop_lr)
                    crop_hr = augment(crop_hr)
                    crop_res_hr = augment(crop_res_hr)
                hr_coord, hr_gt, hr_res = to_pixel_samples(crop_hr.contiguous(), crop_lr, crop_res_hr.contiguous())
                if self.sample_q is not None:
                    sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
                    hr_coord = hr_coord[sample_lst]
                    hr_gt = hr_gt[sample_lst]
                    hr_res = hr_res[sample_lst]
                inp_stack[i] = crop_lr
                coord_stack[i] = hr_coord
                hr_res_stack[i] = hr_res
                gt_stack[i] = hr_gt


        return {
            'inp': inp_stack.float(),
            'coord': coord_stack,
            'hr_res': hr_res_stack,
            'gt': gt_stack.float(),
            'cell': cell_stack
        }

















@register('sr-explicit-downsampled-test-speed')
class MRINewWrapper(Dataset):

    def __init__(self, dataset, out_size, down_scale = None, up_scale = None):
        self.dataset = dataset
        self.down_scale = down_scale
        self.up_scale = up_scale
        self.out_size = out_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ds = self.down_scale
        us = self.up_scale
        img_hr = self.dataset[idx]
        h_lr, w_lr = math.floor(self.out_size / ds + 1e-9), math.floor(self.out_size / ds + 1e-9)
        h_hr, w_hr = int(h_lr * us), int(w_lr * us)
        crop_hr = crop_center(img_hr, self.out_size, self.out_size)
        crop_lr = kspace_zero_padding_resample_torch(crop_hr, h_lr, w_lr)
        # crop_res_hr = resize_image_bicubic(crop_lr, h_hr, w_hr)
        # hr_coord, hr_gt, hr_res = to_pixel_samples(crop_hr.contiguous(), crop_lr, crop_res_hr.contiguous())
        hr_coord = make_coords((h_hr, w_hr), crop_lr.shape[-2 : ])
        # hr_gt = crop_hr.contiguous().view(1, -1).permute(1, 0)
        # hr_res = crop_res_hr.contiguous().view(1, -1).permute(1, 0)
        inp_stack = crop_lr
        coord_stack = hr_coord
        # hr_res_stack = hr_res
        # gt_stack = hr_gt
        cell_stack = torch.tensor([h_lr/h_hr, w_lr/w_hr])
        return {
            'inp': inp_stack.float(),
            'coord': coord_stack,
            # 'hr_res': hr_res_stack,
            # 'gt': gt_stack.float(),
            'cell': cell_stack
        }











@register('sr-explicit-downsampled')
class MRIWrapper(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        if self.inp_size is None:
            h_hr, w_hr = img.shape[-2], img.shape[-1]
            if h_hr > w_hr:
                h_hr = w_hr
                img = img[:, :h_hr, :w_hr]
            elif h_hr < w_hr:
                w_hr = h_hr
                img = img[:, :h_hr, :w_hr]
            h_lr = math.floor(h_hr / s + 1e-9)
            w_lr = math.floor(w_hr / s + 1e-9)
            while w_lr % 4 != 0: w_lr  = w_lr - 1
            while h_lr % 4 != 0: h_lr  = h_lr - 1
            img_down = resize_image_bicubic(img, h_lr, w_lr)#resize_image_bicubic(img, h_lr, w_lr)# #
            crop_lr, crop_hr = img_down, img
            crop_res = resize_image_bicubic(crop_lr, h_hr, w_hr) #kspace_zero_padding_resample_torch(crop_hr, h_lr, w_lr, resize=False)
            cell = torch.tensor([h_lr/h_hr, w_lr/w_hr])
        else:
            h_lr, w_lr = self.inp_size, self.inp_size
            h_hr, w_hr = round(h_lr * s), round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - h_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0 : x0 + h_hr, y0 : y0 + w_hr]
            crop_lr =  resize_image_bicubic(crop_hr, h_lr,w_lr) #resize_image_bicubic(crop_hr, h_lr,w_lr)
            crop_res = resize_image_bicubic(crop_lr, h_hr, w_hr) #kspace_zero_padding_resample_torch(crop_hr, h_lr, w_lr, resize=False)
            cell = torch.tensor([h_lr/h_hr, w_lr/w_hr])

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5
            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            crop_res = augment(crop_res)

        hr_coord, hr_gt, hr_res = to_pixel_samples(crop_hr.contiguous(), crop_lr, crop_res.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_gt = hr_gt[sample_lst]
            hr_res = hr_res[sample_lst]
        
        return {
            'inp': crop_lr.float(),
            'coord': hr_coord,
            'hr_res': hr_res,
            'gt': hr_gt.float(),
            'cell': cell
        }
    
@register('sr-explicit-slice-downsampled')
class MRIWrapper(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        # s = random.randint(self.scale_min + 1, self.scale_max)
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = img.shape[-1]
            img = img[:, :round(h_lr * s), :] # assume round int
            img_down = up_SP_downsample(img, h_lr, round(h_lr * s), 0.625)
            crop_lr, crop_hr = img_down, img
            crop_bicubic = resize_image_bicubic(crop_lr, round(h_lr * s))
            cell = torch.tensor([h_lr/round(h_lr * s), 1])
        else:
            h_lr = self.inp_size
            h_hr = round(h_lr * s)
            x0 = random.randint(0, img.shape[-2] - h_hr)
            y0 = random.randint(0, img.shape[-1] - h_lr)
            crop_hr = img[:, x0 : x0 + h_hr, y0 : y0 + h_lr]
            crop_lr = up_SP_downsample(crop_hr, h_lr, h_hr, 0.625)
            crop_bicubic = resize_image_bicubic(crop_lr, h_hr)
            cell = torch.tensor([h_lr/h_hr, 1])

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            crop_bicubic = augment(crop_bicubic)

        hr_coord, hr_gt, hr_bicubic = to_pixel_samples(crop_hr.contiguous(), crop_lr, crop_bicubic.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_gt = hr_gt[sample_lst]
            hr_bicubic = hr_bicubic[sample_lst]

        
        return {
            'inp': crop_lr.float(),
            'coord': hr_coord,
            'hr_bicubic': hr_bicubic,
            'gt': hr_gt.float(),
            'cell': cell
        }

@register('sr-explicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }