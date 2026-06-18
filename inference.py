import logging
import warnings
from src.utils.io_utils import ROOT_PATH

import hydra
import torch
import numpy as np
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from PIL import Image

from src.datasets.collate import inference_collate_fn

warnings.filterwarnings("ignore", category = UserWarning)

for name in ["httpx", "huggingface_hub", "urllib3", "comet_ml", "datasets"]:
    logging.getLogger(name).setLevel(logging.WARNING)

def normalize_img(img):
    mn = img.amin(dim = (-3, -2, -1), keepdim = True)
    mx = img.amax(dim = (-3, -2, -1), keepdim = True)
    return (img - mn) / (mx - mn + 1e-8)

@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model and
    dataloaders. Runs Inferencer to save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.get("device", "auto") != "auto":
        device = config.device

    model = instantiate(config.model).to(device)

    assert config.get("from_pretrained", None) is not None, "Provide model checkpoint."

    checkpoint = config.get("from_pretrained", None)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)

    print(f"Loaded weights from {checkpoint}")

    model.eval()

    dataset = instantiate(config.datasets)
    if isinstance(dataset, dict):
        dataset = dataset[config.get("partition", "test")]

    loader = DataLoader(
        dataset,
        batch_size = config.dataloader.batch_size,
        num_workers = config.dataloader.get("num_workers", 2),
        collate_fn = inference_collate_fn,
        shuffle = False,
    )

    save_dir = ROOT_PATH / config.get("save_dir", "reconstructions")
    save_dir.mkdir(parents = True, exist_ok = True)

    n = 0
    with torch.no_grad():
        for batch in loader:
            lensless = batch["lensless"].to(device)
            psf = batch["psf"].to(device)
 
            output = model(lensless = lensless, psf = psf)["output"]
            output = normalize_img(output).clamp(0, 1).cpu()
 
            for i, img_id in enumerate(batch["id"]):
                arr = (output[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(arr).save(save_dir / f"{img_id}.png")
                n += 1

    print(f"Saved {n} reconstructions to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
