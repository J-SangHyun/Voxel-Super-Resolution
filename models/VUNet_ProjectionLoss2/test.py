# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from utils.voxel_functions import voxel2obj
from torchmetrics.functional import peak_signal_noise_ratio

from dataloader import ModelNetDataset
from models.VUNet_ProjectionLoss2.VUNet import VUNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Torch device: {device}')

config_files = list((VUNet.path / 'config').glob('*.json'))
print("------- Configuration --------")
for i in range(len(config_files)):
    print(f'{i}. {os.path.basename(config_files[i])}')
config_file = config_files[int(input("Choose Configuration: "))]

with open(config_file, 'r') as f:
    config = json.load(f)

max_epoch = config['max_epoch']
batch_size = config['batch_size']
low_grid, high_grid = config['low_grid'], config['high_grid']

model = VUNet(low_grid, high_grid).to(device)
test_dataset = ModelNetDataset(config['dataset'], 'test', low_grid, high_grid, upsampling=model.need_upsampling)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

alpha = config['alpha']
beta = 1 - alpha

root = Path(os.path.dirname(__file__))
object_dir = root / 'object_examples' / config['dataset'] / f'{low_grid}_{high_grid}_{alpha}'
object_dir.mkdir(parents=True, exist_ok=True)

ckpt_root = model.path / 'ckpt' / config['dataset'] / f'{low_grid}_{high_grid}_{alpha}'
best_path = ckpt_root / 'best.pth'
best_valid_loss = np.inf
last_epoch = 0
if os.path.exists(best_path):
    ckpt = torch.load(best_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model'])
    last_epoch = ckpt['epoch']
    best_val_loss = ckpt['valid_loss']
    print('Best checkpoint is loaded.')
    print(f'Best Epoch: {ckpt["epoch"]} |',
          f'Best Avg Train Loss: {ckpt["train_loss"]} |',
          f'Best Avg Valid Loss: {ckpt["valid_loss"]}')
else:
    print('No checkpoint is found.')

model.eval()
idx = 0
total_psnr = 0.0

with torch.no_grad():
    for lr, hr in test_loader:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr)

        total_psnr += peak_signal_noise_ratio(sr, hr)

        if idx % 100 == 0:
            lr_path = object_dir / f'lr{idx // 100}.obj'
            hr_path = object_dir / f'hr{idx // 100}.obj'
            sr_path = object_dir / f'sr{idx // 100}.obj'

            voxel2obj(lr_path, lr.to('cpu')[0].squeeze(0).detach().numpy())
            voxel2obj(hr_path, hr.to('cpu')[0].squeeze(0).detach().numpy())
            voxel2obj(sr_path, (sr.to('cpu')[0].squeeze(0).detach() > 0.5).float().numpy())
        idx += 1

print(f'AVERAGE PSNR: {total_psnr / idx}')
