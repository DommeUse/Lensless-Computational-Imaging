import logging
import warnings

import hydra
from matplotlib.path import Path
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

import cv2
from PIL import Image

from lensless_helpers.preprocessor import CROPED_LENSED_SHAPE, crop_roi
from lensless_helpers.utils import resize

from src.metrics.tracker import MetricTracker
from torch.utils.data import DataLoader
from src.datasets.collate import collate_fn

from lensless_helpers.preprocessor import crop_roi

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

for name in ["httpx", "huggingface_hub", "comet_ml", "urllib3"]:
    logging.getLogger(name).setLevel(logging.WARNING)

def to_PIL(img, stretch = True):
    img = img.cpu().numpy()
    if stretch:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    else:
        img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img

def normalize_img(img):
    mn = img.amin(dim = (-3, -2, -1), keepdim = True)
    mx = img.amax(dim = (-3, -2, -1), keepdim = True)
    return (img - mn) / (mx - mn + 1e-8)

def evaluate_on_split(config, logger, writer, device, metrics):
    datasets = instantiate(config.datasets)

    tracker = MetricTracker(*([met.name for met in metrics] + ["n_params_admm", "n_params_pre", "n_params_post"]))

    dataloader = DataLoader(
        datasets["test"],
        batch_size = config.dataloader.batch_size,
        num_workers = config.dataloader.num_workers,
        pin_memory = True,
        shuffle = False,
        collate_fn = collate_fn
    )

    model = instantiate(config.model).to(device)

    checkpoint_path = config.get("from_pretrained", None)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = False)
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    logger.info(model)

    n_log_images = config.get("n_log_images", 3)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc = "Calculating metrics on test set")):
            for i in config.trainer.device_tensors:
                batch[i] = batch[i].to(device)

            outputs = model(**batch)
            batch.update(outputs)

            if writer is not None and batch_idx == 0:
                for i in range(min(n_log_images, batch["output"].shape[0])):
                    writer.add_image(
                        f"test_lensless_{i}",
                        to_PIL(batch["lensless"][i], stretch = True)
                    )
                    writer.add_image(
                        f"test_lensed_{i}",
                        to_PIL(batch["lensed"][i], stretch = False)
                    )
                    writer.add_image(
                        f"test_reconstruction_{i}",
                        to_PIL(batch["output"][i], stretch = False)
                    )

            batch["output"] = normalize_img(crop_roi(batch["output"]))
            batch["lensed"] = crop_roi(batch["lensed"])

            for met in metrics:
                tracker.update(met.name, met(**batch), n=batch["output"].shape[0])

    for name in ["admm", "pre", "post"]:
        sub = getattr(model, name, None)
        if sub is not None:
            n_params = sum(p.numel() for p in sub.parameters())
            tracker.update(f"n_params_{name}", n_params, n = 1)

    results = tracker.result()
    print("Metrics on test set:")
    for name, value in results.items():
        print(f"{name}: {value}")
        if writer is not None:
            writer.add_scalar(name, value)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:")
    print(f"total = {total} | trainable = {trainable}")

def load_img(path):
    arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)
 
def collect(dir):
    dir = Path(dir)
    return {p.stem: p for p in sorted(dir.iterdir()) if p.suffix == ".png"}

def to_roi_pair(recon, gt):
    recon_roi = normalize_img(crop_roi(recon)).clamp(0, 1)
 
    if gt.shape[-2:] == recon.shape[-2:]:
        gt_roi = crop_roi(gt)
    else:
        gt_np = resize(gt.permute(1, 2, 0).numpy(), shape = CROPED_LENSED_SHAPE, interpolation = cv2.INTER_NEAREST)
        gt_roi = torch.from_numpy(gt_np).permute(2, 0, 1)

    return recon_roi, gt_roi.clamp(0, 1)

def evaluate_from_dirs(config, logger, writer, device, metrics, gt_dir, recon_dir):
    gt_dir = Path(gt_dir)
    
    tracker = MetricTracker(*[met.name for met in metrics])

    if not gt_dir.is_dir():
        print(f"No ground-truth directory found!")
        return
    
    recon_files = collect(recon_dir)
    gt_files = collect(gt_dir)
    ids = sorted(set(recon_files) & set(gt_files))

    if len(ids) == 0:
        print(f"No matching files found between {gt_dir} and {recon_dir}")
        return
    
    n_log_images = config.get("n_log_images", 3)

    with torch.no_grad():
        for k, img_id in enumerate(tqdm(ids, desc = "Calculating metrics")):
            recon = load_img(recon_files[img_id])
            gt = load_img(gt_files[img_id])
 
            recon_roi, gt_roi = to_roi_pair(recon, gt)
            recon_roi = recon_roi.unsqueeze(0).to(device)
            gt_roi = gt_roi.unsqueeze(0).to(device)
 
            batch = {"output": recon_roi, "lensed": gt_roi}
 
            if writer is not None and k < n_log_images:
                writer.add_image(f"test_reconstruction_{k}", to_PIL(recon_roi[0], stretch = False))
                writer.add_image(f"test_lensed_{k}", to_PIL(gt_roi[0], stretch = False))
 
            for met in metrics:
                tracker.update(met.name, met(**batch), n = batch["output"].shape[0])

    results = tracker.result()
    print(f"Metrics on {len(ids)} image pairs:")
    for name, value in results.items():
        print(f"{name}: {value}")
        if writer is not None:
            writer.add_scalar(name, value)



@hydra.main(version_base=None, config_path="src/configs", config_name="admm100")
def main(config):
    project_config = OmegaConf.to_container(config)

    logger = logging.getLogger("eval")
    writer = None
    if config.get("writer") is not None:
        try:
            writer = instantiate(config.writer, logger, project_config)
            writer.set_step(0, mode = "test")
        except Exception as e:
            logger.warning(f"Failed to initialize writer: {e}")
            writer = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.get("device", "auto") != "auto":
        device = config.device

    metrics = instantiate(config.metrics)["inference"]

    gt_dir = config.get("gt_dir", None)
    recon_dir = config.get("recon_dir", None)

    if gt_dir is not None and recon_dir is not None:
        evaluate_from_dirs(config, logger, writer, device, metrics, gt_dir, recon_dir)
    else:
        evaluate_on_split(config, logger, writer, device, metrics)

if __name__ == "__main__":
    main()
