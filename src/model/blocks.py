import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = n_channels, out_channels = n_channels, kernel_size = (3, 3), padding = 1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = n_channels, out_channels = n_channels, kernel_size = (3, 3), padding = 1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks = 4):
        super().__init__()

        self.res_blocks = nn.Sequential(
            *[ResBlock(in_channels) for _ in range(n_blocks)]
        )

        self.down = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2, stride = 2)

    def forward(self, x):
        out = self.res_blocks(x)
        return self.down(out), out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks = 4):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2, stride = 2)

        self.res_blocks = nn.Sequential(
            *[ResBlock(out_channels) for _ in range(n_blocks)]
        )

    def forward(self, x, skip):
        return self.res_blocks(self.up(x) + skip)

class MidBlock(nn.Module):
    def __init__(self, n_channels, n_blocks = 4):
        super().__init__()

        self.res_blocks = nn.Sequential(
            *[ResBlock(n_channels) for _ in range(n_blocks)]
        )

    def forward(self, x):
        return self.res_blocks(x)