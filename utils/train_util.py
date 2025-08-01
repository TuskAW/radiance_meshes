import torch
import time
import math
import numpy as np
from utils import topo_utils
from icecream import ic
import math
from utils.graphics_utils import l2_normalize_th
from data.camera import focal2fov
from pathlib import Path, PosixPath
import json

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PosixPath):
            return str(o)
        return super().default(o)

class SimpleSampler:
    def __init__(self, total_num_samples, batch_size):
        self.total_num_samples = total_num_samples
        self.batch_size = batch_size
        self.curr = total_num_samples
        self.ids = None

    def nextids(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        self.curr += batch_size
        if self.curr + batch_size > self.total_num_samples:
            # self.ids = torch.LongTensor(np.random.permutation(self.total_num_samples))
            self.ids = torch.randperm(self.total_num_samples, dtype=torch.long, device=device)
            self.curr = 0
        ids = self.ids[self.curr : self.curr + batch_size]
        return ids


class ClippedGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lr_matrix):
        ctx.save_for_backward(lr_matrix)
        return input  # Identity operation

    @staticmethod
    def backward(ctx, grad_output):
        lr_matrix, = ctx.saved_tensors
        grad_norm = torch.linalg.norm(grad_output, dim=-1, keepdim=True)
        # grad_output = torch.maximum(-lr_matrix.abs(), torch.minimum(lr_matrix.abs(), grad_output))
        shape = grad_norm.shape
        clipped_grad_norm = grad_norm.clip(-lr_matrix.abs().reshape(*shape), lr_matrix.abs().reshape(*shape))
        return l2_normalize_th(grad_output) * clipped_grad_norm, None
        # return grad_output, None

class ScaledGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lr_matrix):
        ctx.save_for_backward(lr_matrix)
        return input  # Identity operation

    @staticmethod
    def backward(ctx, grad_output):
        lr_matrix, = ctx.saved_tensors
        return grad_output * lr_matrix, None

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def pad_hw2even(h, w):
    return int(math.ceil(h / 2))*2, int(math.ceil(w / 2))*2

def pad_image2even(im, fnp=np):
    h, w = im.shape[:2]
    nh, nw = pad_hw2even(h, w)
    im_full = fnp.zeros((nh, nw, 3), dtype=im.dtype)
    im_full[:h, :w] = im
    return im_full

class SpikingLR:
    def __init__(self, duration, max_steps, base_function,
                 peak_start, peak_interval, peak_end,
                 peak_lr_init, peak_lr_final):
        self.duration = duration
        self.base_function = base_function
        self.max_steps = max_steps

        self.peak_start = peak_start
        self.peak_interval = peak_interval
        self.peak_end = peak_end

        self.peak_lr_init = peak_lr_init
        self.peak_lr_final = peak_lr_final

    def peak_height_fn(self, i):
        return i / self.max_steps * (self.peak_lr_final - self.peak_lr_init) + self.peak_lr_init
        # return self.peak_lr_init
    
    def peak_fn(self, step, height):
        t = np.clip(step / self.duration, 0, 1)
        log_lerp = np.exp(np.log(height) * (1 - t) + np.log(1e-6) * t)
        return log_lerp
        # return height * math.exp(-step * 6/self.duration + 2/self.duration) / math.exp(2/self.duration)

    def __call__(self, iteration):
        base_f = self.base_function(iteration)
        if self.duration == 0:
            return base_f
        if iteration < self.peak_start:
            return base_f
        elif iteration > self.peak_end:
            last_peak = iteration - self.peak_end
        else:
            last_peak = (iteration - self.peak_start) % self.peak_interval
        peak_ind = iteration - last_peak
        height = self.peak_height_fn(peak_ind) - self.base_function(peak_ind)
        return base_f + self.peak_fn(last_peak, height)

class TwoPhaseLR:
    def __init__(self, max_i, start_i, period_i, settle_i, 
                 lr_peak, lr_end_peak, lr_trough, lr_final):
        self.max_i = max_i
        self.start_i = start_i
        self.settle_i = settle_i
        self.period_i = period_i
        self.lr_peak = lr_peak
        self.lr_end_peak = lr_end_peak
        self.lr_trough = lr_trough
        self.lr_final = lr_final

        n_cycles = settle_i / period_i
        self.gamma = (lr_end_peak / lr_peak) ** (1 / n_cycles) if n_cycles > 0 else 1

    def __call__(self, i):
        # Phase 1: Spiking with decaying cosine annealing
        if i < self.start_i:
            return get_expon_lr_func(self.lr_peak, self.lr_trough, max_steps=self.start_i)(i)
        elif self.start_i <= i <= self.settle_i:
            cycle = math.floor((i-self.start_i) / self.period_i)
            t_cycle = (i-self.start_i) % self.period_i
            
            lr_max = self.lr_peak * (self.gamma ** cycle)
            
            height = (lr_max - self.lr_trough)
            # lr = self.lr_trough + 0.5 * height * \
            #      (1 + math.cos(math.pi * t_cycle / self.period_i))
            t = t_cycle / self.period_i
            lr = self.lr_trough + np.exp(np.log(height) * (1 - t) + np.log(1e-6) * t)
            
            return lr

        # Phase 2: Final settling cosine decay
        else:
            if i >= self.max_i:
                return self.lr_final

            t_settle = i - self.settle_i
            d_settle = self.max_i - self.settle_i
            if d_settle <= 0:
                return self.lr_final
            
            lr = self.lr_final + 0.5 * (self.lr_trough - self.lr_final) * \
                 (1 + math.cos(math.pi * t_settle / d_settle))

            return lr
