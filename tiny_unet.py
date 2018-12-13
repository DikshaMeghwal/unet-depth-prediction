import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.activation(self.conv(out))
        out = self.conv2(out)
        return out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.activation = F.tanh
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block3_16 = UNetConvBlock(3, 16)
        self.conv_block16_32 = UNetConvBlock(16, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)

        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)
        self.up_block32_16 = UNetUpBlock(32, 16)

        self.last = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        block1 = self.conv_block3_16(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block16_32(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block32_64(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block64_128(pool3)

        up1 = self.activation(self.up_block128_64(block4, block3))

        up2 = self.activation(self.up_block64_32(up1, block2))

        up3 = self.up_block32_16(up2, block1)

        return self.last(up3)
