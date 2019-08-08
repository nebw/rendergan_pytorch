import numpy as np
import torch


def normalize_labels(z, param_gen):
    device = z.device
    batch_size = z.shape[0]

    l = param_gen(z)
    
    #z_rot = (l[:, 0] * np.pi)[:, None]
    z_rot = (torch.rand((batch_size, 1), dtype=torch.float32, device=device) * 2 - 1) * np.pi
    y_rot = (l[:, 1] * np.pi / 10)[:, None]
    x_rot = (l[:, 2] * np.pi / 10)[:, None]
    cen = l[:, 3:5] * 5
    rad = ((l[:, 5]) * 2)[:, None] + 1
    
    bits = torch.autograd.Variable(
        torch.from_numpy(np.random.binomial(1, 0.5, (batch_size, 12)).astype(np.float32))).to(device)
    
    l_norm = torch.cat((
        bits,
        torch.cos(z_rot),
        torch.sin(z_rot),
        torch.cos(y_rot),
        torch.sin(y_rot),
        torch.cos(x_rot),
        torch.sin(x_rot),
        rad,
        cen
    ), dim=1)
    
    return l_norm