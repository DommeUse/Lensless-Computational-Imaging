from pathlib import Path
 
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
 
from lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_dataset_object,
)

from lensless_helpers.psf import simulate_psf_from_mask

class CustomDirDataset(Dataset):
    def __init__(self, data_dir, limit = None, *args, **kwargs):
        self.data_dir = Path(data_dir)
        self.lensless_dir = self.data_dir / "lensless"
        self.lensed_dir = self.data_dir / "lensed"
        self.masks_dir = self.data_dir / "masks"

        if not self.lensless_dir.is_dir():
            raise FileNotFoundError(f"Missing required subdir: {self.lensless_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Missing required subdir: {self.masks_dir}")
        

        self.has_lensed = self.lensed_dir.is_dir()

        ids = []

        for p in sorted(self.lensless_dir.iterdir()):
            if p.suffix.lower() == ".png" and (self.masks_dir / f"{p.stem}.npy").exists():
                ids.append(p.stem)

        if limit is not None:
            ids = ids[:limit]

        if len(ids) == 0:
            raise RuntimeError(
                "File format is not supported. Use <id>.png and <id>.npy for lensless and mask images instead."
            )
        
        self.ids = ids
        self._psf_cache = {}
    
    def __len__(self):
        return len(self.ids)
    
    def _find(self, directory, stem):
        p = directory / f"{stem}.png"
        return p if p.exists() else None
    
    def _load_img(path):
        return np.array(Image.open(path).convert("RGB"))
    
    def _get_psf(self, stem, mask):
        if stem not in self._psf_cache:
            self._psf_cache[stem] = simulate_psf_from_mask(mask)
        return self._psf_cache[stem]

    def __getitem__(self, ind):
        stem = self.ids[ind]
 
        lensless_img = self._load_img(self._find(self.lensless_dir, stem))
        mask = np.load(self.masks_dir / f"{stem}.npy")
        psf = self._get_psf(stem, mask)
 
        lensed_path = self._find(self.lensed_dir, stem) if self.has_lensed else None
 
        if lensed_path is not None:
            lensed_img = self._load_img(lensed_path)
            lensed, lensless, psf = get_dataset_object(
                lensed_img, lensless_img, mask, psf = psf
            )

            result = {
                "lensless": self._to_chw(lensless),
                "psf": self._to_chw(psf),
                "lensed": self._to_chw(lensed),
                "id": stem,
            }
        else:
            lensless = convert_image_to_float(force_rgb(lensless_img))
            lensless = torch.rot90(torch.from_numpy(lensless), dims = (-3, -2), k = 2)

            result = {
                "lensless": self._to_chw(lensless),
                "psf": self._to_chw(psf),
                "id": stem,
            }

        return result

    @staticmethod
    def _to_chw(x):
        x = torch.as_tensor(x, dtype = torch.float32)

        if x.dim() == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.dim() == 3 and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)

        return x.contiguous()