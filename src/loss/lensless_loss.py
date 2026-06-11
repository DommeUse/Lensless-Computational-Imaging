import torch
from torch import nn

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class LenslessLoss(nn.Module):
    def __init__(self, device = "auto"):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.loss_mse = nn.MSELoss()
        self.loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type = "vgg", reduction = "mean", normalize = True).to(device)

    def forward(self, output, lensed, **kwargs):
        loss = self.loss_mse(output, lensed) + self.loss_lpips(output.clamp(0, 1), lensed)
        return {"loss": loss}
