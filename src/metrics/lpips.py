import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from src.metrics.base_metric import BaseMetric

class LPIPSMetric(BaseMetric):
    def __init__(self, device = "auto", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = LearnedPerceptualImagePatchSimilarity(net_type = "vgg", reduction = "mean", normalize = True).to(device)

    @torch.no_grad()
    def __call__(self, output, lensed, **kwargs):
        return self.metric(output.clamp(0, 1), lensed).item()
        