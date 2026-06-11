import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.mirflickr_dataset import MirflickrDataset

from pathlib import Path

from src.model.admm import ADMM
from src.model.admm_utils import crop, decrop, H, Ht, Psi, Psit

SPLIT = "test"
LIMIT = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents = True, exist_ok = True)

def to_hwc(img):
    arr = img.detach().cpu().float().permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)

# --- real sample ---
ds = MirflickrDataset(split = SPLIT, limit = LIMIT)
item = ds[0]
lensless = item["lensless"].unsqueeze(0).to(DEVICE)
lensed = item["lensed"].unsqueeze(0).to(DEVICE)
psf = item["psf"].unsqueeze(0).to(DEVICE)
h, w = lensless.shape[-2:]
padded_h, padded_w = h * 2, w * 2
print(f"sample: lensless = {tuple(lensless.shape)} psf = {tuple(psf.shape)} | lensed={tuple(lensed.shape)} | padded = ({padded_h},{padded_w}) | device = {DEVICE}")

# --- check ops ---
admm = ADMM(n_iters = 1, learnable = False).to(DEVICE)
ops = admm._precompute(lensless, psf)

# --- operators test ---
a = torch.randn(1, 3, padded_h, padded_w, device = DEVICE)
b = torch.randn(1, 3, padded_h, padded_w, device = DEVICE)

lhs = torch.sum(H(a, ops["P"]) * b)
rhs = torch.sum(a * Ht(b, ops["P"]))
print("H test:")
print(f"lhs = {lhs} | rhs = {rhs} | diff = {torch.abs(lhs - rhs)}")

a = torch.randn(2, 1, 3, padded_h, padded_w, device = DEVICE)

lhs = torch.sum(Psi(a) * b)
rhs = torch.sum(a * Psit(b))
print("Psi test:")
print(f"lhs = {lhs} | rhs = {rhs} | diff = {torch.abs(lhs - rhs)}")


