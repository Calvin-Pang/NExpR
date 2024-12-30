import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import argparse
from utils import *
import models, math
from tqdm import tqdm
import yaml
import datasets
from torchvision import transforms
from PIL import Image
import warnings
import time
from pytorch_msssim import ssim
from lpips import LPIPS
from FID.fid_score import FID
import sys
from datasets.wrapper import kspace_zero_padding_resample_torch
warnings.filterwarnings("ignore")
def make_data_loader(spec):
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
            shuffle=False, num_workers=8, pin_memory=True)
    return loader
def find_keys(d, prefix='val_dataset_x'):
    matching_keys = []
    for key in d:
        if key.startswith(prefix):
            matching_keys.append(key)
        if isinstance(d[key], dict):
            matching_keys.extend(find_keys(d[key], prefix))
    return matching_keys

def test(model, test_loaders, loss_fn_lpips, val_img_dir):
    model.eval()

    with torch.no_grad():
        for test_loader_pair in test_loaders:
            scale, test_loader = test_loader_pair
            num = len(test_loader)
            log('test sr_x' + str(scale))
            scale_dir = os.path.join(val_img_dir, 'sr_x' + str(scale))
            if not(os.path.exists(scale_dir)): os.makedirs(scale_dir)
            gt_dir = os.path.join(scale_dir, 'gt')
            if not(os.path.exists(gt_dir)): os.makedirs(gt_dir)
            lr_dir = os.path.join(scale_dir, 'lr')
            if not(os.path.exists(lr_dir)): os.makedirs(lr_dir)
            inte_dir = os.path.join(scale_dir, 'inte')
            if not(os.path.exists(inte_dir)): os.makedirs(inte_dir)
            res_dir = os.path.join(scale_dir, 'res')
            if not(os.path.exists(res_dir)): os.makedirs(res_dir)
            sr_dir = os.path.join(scale_dir, 'sr')
            if not(os.path.exists(sr_dir)): os.makedirs(sr_dir)
            psnr_avg = 0
            ssim_avg = 0
            ssim_inte = 0
            psnr_inte = 0
            psnr_con = 0
            time_used = 0
            lpips_loss = 0
            lpips_inte = 0
            psnr_con_inte = 0
            cnt = 0
            for sample in tqdm(test_loader, leave = False, desc ='validating...'):
                for k, v in sample.items():
                    sample[k] = v.cuda()
                gt = sample['gt']
                lr = sample['inp']
                ts_res = sample['hr_res']
                coord_hr = sample['coord']
                cnt += 1
                _, C, dsample_h, dsample_w = lr.shape
                # H, W = round(dsample_h*scale), round(dsample_w*scale)
                H, W = int(math.sqrt(len(gt[0]))), int(math.sqrt(len(gt[0])))
                cell = sample['cell']
                img_res = transforms.ToPILImage()(ts_res.view(C,H,W))
                time_s = time.time()
                pred_intensity = model(lr, coord_hr, cell)

                # save_dict = {}
                # save_dict['x_freq'] = model.x_coord
                # save_dict['y_freq'] = model.y_coord
                # save_dict['map'] = model.function_map
                # torch.save(save_dict, './functions.pth')
                # sys.exit()
                time_e = time.time()
                time_used += time_e - time_s
                pred_intensity = pred_intensity + ts_res
                pred_intensity = pred_intensity.reshape(C,H,W).clamp(0, 1) # (1, H, W)
                             
                sr_ds_ts = transforms.ToTensor()(transforms.Resize((dsample_h, dsample_w), 
                                                                   Image.Resampling.BICUBIC)(transforms.ToPILImage()(pred_intensity)))
                sr_inte = ts_res.reshape(C, H, W)
                psnr_con_sample = calc_psnr(sr_ds_ts.cuda(), lr)
                psnr_con += psnr_con_sample

                inte_ds_ts = transforms.ToTensor()(transforms.Resize((dsample_h, dsample_w), 
                                                                     Image.Resampling.BICUBIC)(transforms.ToPILImage()(sr_inte)))
                psnr_con_inte_sample = calc_psnr(inte_ds_ts.cuda(), lr)
                psnr_con_inte += psnr_con_inte_sample
                
                psnr_inte_sample = calc_psnr(sr_inte, gt.view(C,H,W))
                psnr_inte += psnr_inte_sample
                ssim_inte += ssim(sr_inte.view(1,C,H,W), gt.view(1,C,H,W), data_range=1.0)
                psnr_avg += calc_psnr(pred_intensity, gt.view(C,H,W))
                ssim_avg += ssim(pred_intensity.unsqueeze(0), gt.view(1,C,H,W), data_range=1.0)
                lpips_loss += loss_fn_lpips(pred_intensity, gt.view(C,H,W))
                lpips_inte += loss_fn_lpips(sr_inte, gt.view(C,H,W))
                gt_img = transforms.ToPILImage()(gt.view(C,H,W))
                sr_img = transforms.ToPILImage()(pred_intensity) 
                ds_repeat = transforms.Resize((H,W), Image.Resampling.NEAREST)(transforms.ToPILImage()(lr[0]))
                ds_repeat.save(os.path.join(lr_dir, str(cnt)+'_lr.png'))
                gt_img.save(os.path.join(gt_dir, str(cnt)+'_gt.png'))
                transforms.ToPILImage()(sr_inte).save(os.path.join(inte_dir, str(cnt)+'_inte.png'))
                img_res.save(os.path.join(res_dir, str(cnt)+'_res.png'))
                sr_img.save(os.path.join(sr_dir, str(cnt)+'_sr.png'))
            psnr_avg, ssim_avg = psnr_avg / num, ssim_avg / num
            psnr_inte, ssim_inte = psnr_inte / num, ssim_inte / num
            psnr_con = psnr_con / num
            psnr_con_inte = psnr_con_inte / num
            lpips_loss = lpips_loss / num
            lpips_inte = lpips_inte / num
            time_avg = (time_used / num) * 1000
            log('test: psnr_inte = {:.5f}'.format(psnr_inte))
            log('test: psnr = {:.5f}'.format(psnr_avg))
            log('test: ssim_inte = {:.5f}'.format(ssim_inte))
            log('test: ssim = {:.5f}'.format(ssim_avg))
            log('test: lpips = {:.5f}'.format(lpips_loss.item()))
            log('test: lpips_inte = {:.5f}'.format(lpips_inte.item()))
            log('time per slice: time = {:.5f}'.format(time_avg))
            log('test: consistency = {:.5f}'.format(psnr_con))
            log('test: consistency_inte = {:.5f}'.format(psnr_con_inte))
            
            print()


def main(model, val_loaders, config_, loss_fn_lpips, save_path):
    global config, log
    config = config_
    log = set_test_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print('testing begins...')
    print()
    test(model, val_loaders, loss_fn_lpips, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu')
    parser.add_argument('--name')
    parser.add_argument('--dir')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
        print('config loaded.')
    model = models.make(config['model']).cuda()

    checkpoint = torch.load(config['checkpoint_dir'])
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.cal_x_y_coord()
    print('Checkpoint loaded!')
    val_scales = find_keys(config)
    val_loaders = []
    for val_scale in val_scales:
        scale = int(val_scale[-1])
        val_loader = make_data_loader(config.get(val_scale))
        val_loaders.append((scale, val_loader))

    name = args.name
    save_path = os.path.join('.',args.dir, name)
    print('device:',torch.cuda.current_device())
    loss_fn_lpips = LPIPS(net='vgg').cuda()
    main(model, val_loaders, config, loss_fn_lpips, save_path)