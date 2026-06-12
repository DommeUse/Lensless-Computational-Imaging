import logging
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from src.metrics.tracker import MetricTracker
from torch.utils.data import DataLoader
from src.datasets.collate import collate_fn

from lensless_helpers.preprocessor import crop_roi

warnings.filterwarnings("ignore", category=UserWarning)


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
            writer = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.get("device", "auto") != "auto":
        device = config.device

    datasets = instantiate(config.datasets)
    metrics = instantiate(config.metrics)["inference"]
    tracker = MetricTracker(*[met.name for met in metrics])

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
                        batch["lensless"][i].cpu().permute(1, 2, 0).numpy()
                    )
                    writer.add_image(
                        f"test_lensed_{i}",
                        batch["lensed"][i].cpu().permute(1, 2, 0).numpy()
                    )
                    writer.add_image(
                        f"test_reconstruction_{i}",
                        batch["output"][i].cpu().permute(1, 2, 0).numpy()
                    )

            batch["output"] = crop_roi(batch["output"])
            batch["lensed"] = crop_roi(batch["lensed"])

            for met in metrics:
                tracker.update(met.name, met(**batch), n=batch["output"].shape[0])

    results = tracker.result()
    print("Metrics on test set:")
    for name, value in results.items():
        print(f"{name}: {value}")
        if writer is not None:
            writer.log_metric(name, value, step = 0)

if __name__ == "__main__":
    main()
