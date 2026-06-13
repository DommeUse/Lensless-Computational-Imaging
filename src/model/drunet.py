import torch.nn as nn
import torch
import torch.nn.functional as F

from src.model.blocks import DownBlock, UpBlock, MidBlock


class DRUNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_blocks = 4):
        super().__init__()

        self.input_conv = nn.Conv2d(in_channels = in_channels, out_channels = hidden_channels, kernel_size = 3, padding = 1)

        self.down_block1 = DownBlock(in_channels = 1 * hidden_channels, out_channels = 2 * hidden_channels, n_blocks = n_blocks)
        self.down_block2 = DownBlock(in_channels = 2 * hidden_channels, out_channels = 4 * hidden_channels, n_blocks = n_blocks)
        self.down_block3 = DownBlock(in_channels = 4 * hidden_channels, out_channels = 8 * hidden_channels, n_blocks = n_blocks)

        self.mid_block = MidBlock(n_channels = 8 * hidden_channels, n_blocks = n_blocks)

        self.up_block1 = UpBlock(in_channels = 8 * hidden_channels, out_channels = 4 * hidden_channels, n_blocks = n_blocks)
        self.up_block2 = UpBlock(in_channels = 4 * hidden_channels, out_channels = 2 * hidden_channels, n_blocks = n_blocks)
        self.up_block3 = UpBlock(in_channels = 2 * hidden_channels, out_channels = 1 * hidden_channels, n_blocks = n_blocks)

        self.output_conv = nn.Conv2d(in_channels = hidden_channels, out_channels = in_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % 8, (-w) % 8
        x = F.pad(x, (0, pad_w, 0, pad_h))

        out = self.input_conv(x)

        out, skip1 = self.down_block1(out)
        out, skip2 = self.down_block2(out)
        out, skip3 = self.down_block3(out)

        out = self.mid_block(out)

        out = self.up_block1(out, skip3)
        out = self.up_block2(out, skip2)
        out = self.up_block3(out, skip1)

        out = self.output_conv(out)

        return out[..., :h, :w]