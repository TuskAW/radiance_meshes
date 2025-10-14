#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from icecream import ic
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def depth_to_normals(depth, fx, fy):
    """Assuming `depth` is orthographic, linearize it to a set of normals."""

    f_blur = torch.tensor([1, 2, 1], device=depth.device) / 4
    f_edge = torch.tensor([-1, 0, 1], device=depth.device) / 2
    depth = depth.unsqueeze(0).unsqueeze(0).squeeze(-1)
    dy = F.conv2d(
        depth, (f_blur[None, :] * f_edge[:, None]).unsqueeze(0).unsqueeze(0), padding=1
    )[0, 0]
    dx = F.conv2d(
        depth, (f_blur[:, None] * f_edge[None, :]).unsqueeze(0).unsqueeze(0), padding=1
    )[0, 0]

    # so dx, dy are in image space but we want to transform them to world space
    dx = dx * fx * 2 / depth[0, 0]
    dy = dy * fy * 2 / depth[0, 0]
    inv_denom = 1 / torch.sqrt(1 + dx**2 + dy**2)
    normals = torch.stack([dx * inv_denom, -dy * inv_denom, inv_denom], -1)
    return normals

def depth_to_camera_normals(depth, fx, fy):
    """Calculates normals in camera space from an orthographic depth map."""

    f_blur = torch.tensor([1, 2, 1], device=depth.device, dtype=torch.float32) / 4
    f_edge = torch.tensor([-1, 0, 1], device=depth.device, dtype=torch.float32) / 2
    
    # Reshape for convolution
    depth = depth.unsqueeze(0).unsqueeze(0)

    # Sobel filters to get gradients
    dy = F.conv2d(
        depth, (f_blur[None, :] * f_edge[:, None]).unsqueeze(0).unsqueeze(0), padding='same'
    )[0, 0]
    dx = F.conv2d(
        depth, (f_blur[:, None] * f_edge[None, :]).unsqueeze(0).unsqueeze(0), padding='same'
    )[0, 0]

    # The derivatives dx and dy are in pixel units (change in depth per pixel).
    # We convert them to camera space units.
    # Note: Using per-pixel depth is more accurate than a single depth value.
    depth_val = depth.squeeze().clip(min=1e-6)

    # Convert gradients to camera space
    dx_cam = dx * fx / depth_val
    dy_cam = dy * fy / depth_val

    # Construct normals in camera space
    # The vector is [-dx, -dy, 1] to account for image Y-down and camera Y-up conventions
    # and to have the normal point towards the camera in a right-handed system (-Z view).
    inv_denom = 1 / torch.sqrt(1 + dx_cam**2 + dy_cam**2)
    normals_camera = torch.stack([-dx_cam * inv_denom, -dy_cam * inv_denom, inv_denom], -1)
    
    return normals_camera


def calculate_norm_loss(xyzd, fx, fy):
    pred_normals = depth_to_normals(xyzd[..., 3], fx, fy)
    field_normals = xyzd[..., :3]
    align_world_loss = 2 * (
        1 - (pred_normals * field_normals).sum(dim=-1)
    )
    return align_world_loss.mean()
