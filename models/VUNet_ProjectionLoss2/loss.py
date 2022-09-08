# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def projection_loss(input, target):
    device = input.device
    B, C, D, H, W = input.size()
    input_light_from_D = torch.ones(B, C, H, W, device=device)
    input_light_from_H = torch.ones(B, C, D, W, device=device)
    input_light_from_W = torch.ones(B, C, D, H, device=device)
    target_light_from_D = torch.ones(B, C, H, W, device=device)
    target_light_from_H = torch.ones(B, C, D, W, device=device)
    target_light_from_W = torch.ones(B, C, D, H, device=device)

    for d in range(D):
        input_light_from_D = input_light_from_D * (1 - input[:, :, d, :, :])
        target_light_from_D = target_light_from_D * (1 - target[:, :, d, :, :])

    for h in range(H):
        input_light_from_H = input_light_from_H * (1 - input[:, :, :, h, :])
        target_light_from_H = target_light_from_H * (1 - target[:, :, :, h, :])

    for w in range(W):
        input_light_from_W = input_light_from_W * (1 - input[:, :, :, :, w])
        target_light_from_W = target_light_from_W * (1 - target[:, :, :, :, w])

    loss = (F.mse_loss(input_light_from_D, target_light_from_D) +
            F.mse_loss(input_light_from_H, target_light_from_H) +
            F.mse_loss(input_light_from_W, target_light_from_W)) / 3

    return loss
