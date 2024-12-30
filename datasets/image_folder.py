import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from datasets import register
from PIL import Image
from utils import *
@register('image-folder')
class RGBImageFolder(Dataset):
    def __init__(self, root_path, repeat=1):
        self.repeat = repeat
        filenames = sorted(os.listdir(root_path))
        self.files = []
        for filename in tqdm(filenames, leave = False, desc = 'loading data...'):
            data = Image.open(os.path.join(root_path, filename)) 
            self.files.append(transforms.ToTensor()(data))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        return x


@register('npy-folder')
class MRIImageFolder(Dataset):
    def __init__(self, root_path, first_k = None, repeat=1, normalization=True):
        self.repeat = repeat
        filenames = sorted(os.listdir(root_path))
        if first_k is not None:
            filenames = filenames[:first_k]
        self.files = []
        for filename in tqdm(filenames, leave = False, desc = 'loading data...'):
            data = np.load(os.path.join(root_path, filename)) # (S, 320, 320)
            if normalization:
                min_data = np.min(data)
                max_data = np.max(data)
                data = (data - min_data) / (max_data - min_data) 
            for slice in data:
                self.files.append(transforms.ToTensor()(slice.astype(np.float64)))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        return x
    
@register('npy-folder-new')
class MRINewImageFolder(Dataset):
    def __init__(self, root_path, out_size, first_k = None, normalization=True):
        filenames = sorted(os.listdir(root_path))
        if first_k is not None:
            filenames = filenames[:first_k]
        self.files = []
        for filename in tqdm(filenames, leave = False, desc = 'loading data...'):
            data = np.load(os.path.join(root_path, filename)) # (S, 320, 320)
            if normalization:
                min_data = np.min(data)
                max_data = np.max(data)
                data = (data - min_data) / (max_data - min_data) 
            for slice in data:
                self.files.append(resize_img_to_out_size(transforms.ToTensor()(slice.astype(np.float64)), out_size))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]
        return x

@register('npy-folder-ct')
class MRINewImageFolderCT(Dataset):
    def __init__(self, root_path, out_size, first_k = None, normalization=False):
        filenames = sorted(os.listdir(root_path))
        if first_k is not None:
            filenames = filenames[:first_k]
        self.files = []
        for filename in tqdm(filenames, leave = False, desc = 'loading data...'):
            data = np.load(os.path.join(root_path, filename)) # (S, 320, 320)
            if normalization:
                min_data = np.min(data)
                max_data = np.max(data)
                data = (data - min_data) / (max_data - min_data) 
            for slice in data:
                resized_slice = transforms.ToTensor()(transforms.Resize((out_size, out_size), 
                                                                        Image.Resampling.BICUBIC)(transforms.ToPILImage()(transforms.ToTensor()(slice.astype(np.float64)))))
                self.files.append(resized_slice)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]
        return x








@register('paired-npy-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = MRIImageFolder(root_path_1, **kwargs)
        self.dataset_2 = MRIImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
