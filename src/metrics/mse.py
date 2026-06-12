import torch
from torchmetrics import MeanSquaredError
from src.metrics.base_metric import BaseMetric

class MSEMetric(BaseMetric):
    def __init__(self, device = "auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = MeanSquaredError().to(device)

    @torch.no_grad()
    def __call__(self, output, lensed, **kwargs):
        return self.metric(output.contiguous(), lensed.contiguous()).item()
        