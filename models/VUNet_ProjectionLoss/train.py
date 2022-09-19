# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import ModelNetDataset
from models.VUNet_ProjectionLoss.VUNet import VUNet
from models.VUNet_ProjectionLoss.loss import projection_loss

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
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
train_dataset = ModelNetDataset(config['dataset'], 'train', low_grid, high_grid, upsampling=model.need_upsampling)
valid_dataset = ModelNetDataset(config['dataset'], 'valid', low_grid, high_grid, upsampling=model.need_upsampling)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=1)

alpha = config['alpha']
beta = 1 - alpha

ckpt_root = model.path / 'ckpt' / config['dataset'] / f'{low_grid}_{high_grid}_{alpha}'
ckpt_root.mkdir(parents=True, exist_ok=True)
last_path = ckpt_root / 'last.pth'
best_path = ckpt_root / 'best.pth'
log_dir = ckpt_root / 'log'
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir)
best_valid_loss = np.inf
last_epoch = 0

if os.path.exists(best_path):
    ckpt = torch.load(best_path)
    best_valid_loss = ckpt['valid_loss']

if os.path.exists(last_path):
    ckpt = torch.load(last_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    last_epoch = ckpt['epoch']
    print('Last checkpoint is loaded.')
    print(f'Last Epoch: {ckpt["epoch"]} |',
          f'Last Avg Train Loss: {ckpt["train_loss"]} |',
          f'Last Avg Valid Loss: {ckpt["valid_loss"]}')
else:
    print('No checkpoint is found.')

for epoch in range(last_epoch+1, max_epoch+1):
    start_epoch = time.time()
    print(f'-------- EPOCH {epoch} / {max_epoch} --------')

    model.train()
    train_iter, train_loss = 0, 0
    for lr, hr in train_loader:
        lr = lr.to(device)
        hr = hr.to(device)
        sr = model(lr)

        loss = alpha * F.mse_loss(hr, sr) + beta * projection_loss(hr, sr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iter += 1

    model.eval()
    valid_iter, valid_loss = 0, 0
    with torch.no_grad():
        for lr, hr in valid_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)

            loss = alpha * F.mse_loss(hr, sr) + beta * projection_loss(hr, sr)

            valid_loss += loss.item()
            valid_iter += 1

    avg_train_loss = train_loss / len(train_loader)
    avg_valid_loss = valid_loss / len(valid_loader)
    print(f'EPOCH {epoch}/{max_epoch} |',
          f'Avg Train Loss: {avg_train_loss} |',
          f'Avg Valid Loss: {avg_valid_loss}')
    print(f'This epoch took {time.time() - start_epoch} seconds')

    writer.add_scalar('train_loss', avg_train_loss, epoch)
    writer.add_scalar('valid_loss', avg_valid_loss, epoch)
    writer.flush()

    ckpt = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss}
    torch.save(ckpt, last_path)

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        torch.save(ckpt, best_path)
        print('New Best Model!')
