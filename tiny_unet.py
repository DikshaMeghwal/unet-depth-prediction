import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=2 ,activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1, stride=2)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=2, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.activation = activation

    def forward(self, x):
        up = self.up(x)
        out = self.activation(up)
        return out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.activation = F.relu

        self.conv_block3_16 = UNetConvBlock(3, 16)
        self.conv_block16_32 = UNetConvBlock(16, 32)

        self.up_block32_16 = UNetUpBlock(32, 16)
        self.up_block16_1 = UNetUpBlock(16, 1)

    def forward(self, x):
        block1 = self.conv_block3_16(x)
        block2 = self.conv_block16_32(block1)
        up1 = self.up_block32_16(block2)
        up2 = self.up_block16_1(up1)
        return up2

