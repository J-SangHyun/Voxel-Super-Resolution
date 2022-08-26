# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def projection_loss(input, target):
    assert len(input.size()) == 5

    loss = 0.0
    for dim in range(2, 5):
        grid_size = input.size()[dim]
        input_projection = torch.sum(input, dim=dim)
        target_projection = torch.sum(target, dim=dim)
        projection_mse_loss = F.mse_loss(input_projection, target_projection) / grid_size
        loss += projection_mse_loss
    return loss / 3
