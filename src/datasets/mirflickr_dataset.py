from huggingface_hub import hf_hub_download
from datasets import load_dataset
import numpy as np
import torch

from src.datasets.base_dataset import BaseDataset
from lensless_helpers.preprocessor import get_dataset_object

REPO_URL = "bezzam/DigiCam-Mirflickr-MultiMask-10K"

class MirflickrDataset(BaseDataset):
    def __init__(self, split, limit = None, *args, **kwargs):
        self.hf_dataset = load_dataset(REPO_URL, split = split)
        self.mask_cache = {}

        index = [{"id" : i, "mask_label": self.hf_dataset[i]["mask_label"]} for i in range(len(self.hf_dataset))]

        super().__init__(index, limit = limit, *args, **kwargs)

    def _get_mask(self, mask_label):
        if mask_label not in self.mask_cache:
            path = hf_hub_download(REPO_URL, f"masks/mask_{mask_label}.npy", repo_type = "dataset")
            self.mask_cache[mask_label] = np.load(path)
        return self.mask_cache[mask_label]

    def __getitem__(self, ind):
        element = self._index[ind]
        mask_label = element["mask_label"]
        row = self.hf_dataset[element["id"]]

        mask = self._get_mask(mask_label)
        lensed, lensless, psf = get_dataset_object(
            row["lensed"], 
            row["lensless"], 
            mask
        )

        result = {
            "lensed": self._to_chw(lensed),
            "lensless": self._to_chw(lensless),
            "psf": self._to_chw(psf),
            "id": element["id"],
            "mask_label": mask_label
        }

        return self.preprocess_data(result)
    
    @staticmethod
    def _to_chw(x):
        x = torch.as_tensor(x, dtype = torch.float32)

        if x.dim() == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.dim() == 3 and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)

        return x.contiguous()

    @staticmethod
    def _assert_index_is_valid(index):
        for element in index:
            if "mask_label" not in element:
                raise ValueError(f"Index element should contain 'mask_label' key")
            if "id" not in element:
                raise ValueError(f"Index element should contain 'id' key")