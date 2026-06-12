import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from src.metrics.base_metric import BaseMetric

class PSNRMetric(BaseMetric):
    def __init__(self, device = "auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = PeakSignalNoiseRatio(data_range = (0, 1), reduction = "elementwise_mean").to(device)

    @torch.no_grad()
    def __call__(self, output, lensed, **kwargs):
        return self.metric(output.clamp(0, 1).contiguous(), lensed.contiguous()).item()
        