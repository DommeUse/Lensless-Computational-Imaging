import torch.nn as nn

class ModularReconstruction(nn.Module):
    def __init__(self, admm, pre = None, post = None):
        super().__init__()
        self.pre = pre
        self.admm = admm
        self.post = post

    def forward(self, lensless, psf, **kwargs):
        if self.pre is not None:
            lensless = self.pre(lensless)

        recon = self.admm(lensless, psf)["output"]

        if self.post is not None:
            recon = self.post(recon)
            
        return {"output": recon}