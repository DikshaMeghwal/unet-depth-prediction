import torch
import torch.nn as nn

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, padding=1, stride=2):
        super(UNetConvBlock, self).__init__()
        self.down = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        return self.down(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=2, stride=2):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.up(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.activation = nn.ReLU()

        self.down_block3_16 = UNetConvBlock(3, 16)
        self.down_block16_32 = UNetConvBlock(16, 32)
        self.down_block32_64 = UNetConvBlock(32, 64)

        self.up_block64_32 = UNetUpBlock(64, 32)
        self.up_block32_16 = UNetUpBlock(32, 16)
        self.up_block16_1 = UNetUpBlock(16, 1)

    def forward(self, x):
        x = self.activation(self.down_block3_16(x))
        x = self.activation(self.down_block16_32(x))
        x = self.activation(self.down_block32_64(x))
        
        x = self.activation(self.up_block64_32(x))
        x = self.activation(self.up_block32_16(x))
        x = self.up_block16_1(x)
        return x
