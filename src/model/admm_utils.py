import torch

def crop(x, h, w):
    padded_h, padded_w = x.shape[-2:]
    start_h = (padded_h - h) // 2
    start_w = (padded_w - w) // 2
    return x[..., start_h : start_h + h, start_w : start_w + w]

def decrop(x, padded_h, padded_w):
    h, w = x.shape[-2:]
    start_h = (padded_h - h) // 2
    start_w = (padded_w - w) // 2
    out = torch.zeros(*x.shape[:-2], padded_h, padded_w, device = x.device, dtype = x.dtype)
    out[..., start_h : start_h + h, start_w : start_w + w] = x
    return out

def H(x, P):
    X = torch.fft.rfft2(x)
    return torch.fft.irfft2(P * X, s = x.shape[-2:])

def Ht(x, P):
    X = torch.fft.rfft2(x)
    return torch.fft.irfft2(P.conj() * X, s = x.shape[-2:])

def Psi(x):
    return torch.stack(
        [torch.roll(x, shifts = -1, dims = -1) - x, torch.roll(x, shifts = -1, dims = -2) - x],
        dim = 0
    )

def Psit(x):
    return (torch.roll(x[0], shifts = 1, dims = -1) - x[0]) + (torch.roll(x[1], shifts = 1, dims = -2) - x[1])

def soft_thresholding(x, threshold):
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))