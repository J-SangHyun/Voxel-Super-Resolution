# -*- coding: utf-8 -*-
import torch
from pathlib import Path
import scipy.ndimage as nd
from torch.utils.data import Dataset
from utils.voxel_functions import binvox2numpy


class ModelNetDataset(Dataset):
    def __init__(self, dataset, mode, lr, hr, upsampling=False):
        super(ModelNetDataset, self).__init__()
        self.lr, self.hr = lr, hr
        self.rate = hr // lr
        self.dataset = dataset
        self.mode = mode
        self.upsampling = upsampling

        assert dataset in ['ModelNet10', 'ModelNet40']
        assert mode in ['train', 'val', 'test']
        assert hr > lr and hr % lr == 0

        self.lr_files, self.hr_files = [], []
        lr_classes = Path(f'voxelized/{lr}/{dataset}/').glob('*')
        hr_classes = Path(f'voxelized/{hr}/{dataset}/').glob('*')
        for lr_class in lr_classes:
            if mode in ['train', 'val']:
                train_files = list((lr_class / 'train').glob('*.binvox'))
                if mode == 'train':
                    self.lr_files += train_files[:-len(train_files)//10]
                elif mode == 'val':
                    self.lr_files += train_files[-len(train_files)//10:]
            elif mode == 'test':
                self.lr_files += list((lr_class / 'test').glob('*.binvox'))

        for hr_class in hr_classes:
            if mode in ['train', 'val']:
                train_files = list((hr_class / 'train').glob('*.binvox'))
                if mode == 'train':
                    self.hr_files += train_files[:-len(train_files)//10]
                elif mode == 'val':
                    self.hr_files += train_files[-len(train_files)//10:]
            elif mode == 'test':
                self.hr_files += list((hr_class / 'test').glob('*.binvox'))

        assert len(self.lr_files) == len(self.hr_files)

    def __getitem__(self, idx):
        lr = binvox2numpy(self.lr_files[idx])
        if self.upsampling:
            lr = nd.zoom(lr, (self.rate, self.rate, self.rate), mode='constant', order=0)
        hr = binvox2numpy(self.hr_files[idx])
        return torch.Tensor(lr), torch.Tensor(hr)

    def __len__(self):
        return len(self.lr_files)
