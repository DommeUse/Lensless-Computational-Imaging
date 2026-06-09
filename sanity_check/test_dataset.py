import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.collate import collate_fn
from src.datasets.mirflickr_dataset import MirflickrDataset
from lensless_helpers.preprocessor import get_roi

SPLIT = "test"
LIMIT = 4
BATCH_SIZE = 2


def to_hwc(img):
    arr = img.detach().cpu().float().permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)

# --- building dataset ---
ds = MirflickrDataset(split = SPLIT, limit = LIMIT)
print(f"len(dataset) = {len(ds)}")

# --- observe single item ---
item = ds[0]
print(f"keys = {sorted(item.keys())}")
print(f"id = {item['id']}")
for key in ["lensless", "psf", "lensed"]:
    t = item[key]
    print(f"{key} shape = {tuple(t.shape)}, dtype = {t.dtype}")

# --- ROI check ---
roi = get_roi(item["lensed"].permute(1, 2, 0))
print(f"lensed {tuple(item['lensed'].shape)} -> roi {tuple(roi.shape)}")

# --- batching ---
loader = DataLoader(ds, batch_size = BATCH_SIZE, shuffle = False, collate_fn = collate_fn)
batch = next(iter(loader))
for key in ["lensless", "psf", "lensed"]:
    print(f"batched {key} {batch[key].shape}")
print(f"ids = {batch['id']}")

# --- visualization ---
psf = item["psf"]

fig, axes = plt.subplots(1, 3, figsize = (12, 4))
axes[0].imshow(to_hwc(item["lensless"]))
axes[0].set_title("lensless")

axes[1].imshow(to_hwc(item["lensed"]))
axes[1].set_title("lensed")

axes[2].imshow(to_hwc(psf))
axes[2].set_title("psf")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
fig.savefig("tmp/dataset_check.png", dpi = 120)

print("Successfully saved -> tmp/dataset_check.png")