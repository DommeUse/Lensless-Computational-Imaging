import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from src.metrics.base_metric import BaseMetric

class SSIMMetric(BaseMetric):
    def __init__(self, device = "auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = StructuralSimilarityIndexMeasure(data_range = (0, 1)).to(device)

    @torch.no_grad()
    def __call__(self, output, lensed, **kwargs):
        return self.metric(output, lensed).item()
        