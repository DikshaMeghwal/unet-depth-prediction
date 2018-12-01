import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class UNet(nn.Module):
    def __init__(self, init_weights=True):
        super(UNet, self).__init__()                                   #252 * 252 * 3 - input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)                   #250 * 250 * 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)                  #248 * 248 * 64
                                                                       #after maxpool 124 * 124 * 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)                 #122 * 122 * 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)                #120 * 120 * 128
                                                                       #after maxpool 60 * 60 * 128
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)                #58 * 58 * 256
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)                #56 * 56 * 256
                                                                       #after maxpool 28 * 28 * 256
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3)                #26 * 26 * 512
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3)                #24 * 24 * 512
                                                                       #after maxpool 12 * 12 * 512
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3)               #10 * 10 * 1024
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3)             #8 * 8 * 1024

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')      #16 * 16 * 1024  
        self.upConv1 = nn.Conv2d(1024, 512, kernel_size=1)               #16 * 16 * 512
                                                                         #after concatenation with crop of 16 * 16 * 512
                                                                         #16 * 16 * 1024
        self.deConv1 = nn.Conv2d(1024, 512, kernel_size = 3)             #14 * 14 * 512
        #self.conv8                                                      #12 * 12 * 512

        #upsample 1                                                      #24 * 24 * 512
        self.upConv2 = nn.Conv2d(512, 256, kernel_size=1)                #24 * 24 * 256
                                                                         #after concatenation with crop of 24 * 24 * 256
                                                                         #24 * 24 * 512 
        self.deConv2 = nn.Conv2d(512, 256, kernel_size = 3)              #22 * 22 * 256
        #self.conv6                                                      #20 * 20 * 256

        #upsample 1                                                      #40 * 40 * 256
        self.upConv3 = nn.Conv2d(256, 128, kernel_size=1)                #40 * 40 * 128
                                                                         #after concatenation with crop of 40 * 40 * 128
                                                                         #40 * 40 * 256 
        self.deConv3 = nn.Conv2d(256, 128, kernel_size = 3)              #38 * 38 * 128
        #self.conv4                                                      #36 * 36 * 128

        #upsample 1                                                      #72 * 37 * 128
        self.upConv4 = nn.Conv2d(128, 64, kernel_size=1)                 #72 * 72 * 64
                                                                         #after concatenation with crop of 72 * 72 * 64
                                                                         #72 * 72 * 128
        self.deConv4 = nn.Conv2d(128, 64, kernel_size = 3)               #70 * 70 * 64
        #self.conv2                                                      #68 * 68 * 64
        self.deConv5 = nn.Conv2d(64, 1, kernel_size = 1)                 #68 * 68 * 1

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
                                                                         #H * W * c
                                                                         #252 * 252 * 3
        out1 = F.relu(self.conv1(x))                                     #250 * 250 * 64
        #print('out1:{}'.format(out1.shape))
        out2 = F.relu(self.conv2(out1))                                  #248 * 248 * 64
        #print('out2:{}'.format(out2.shape))
        out3 = self.pool(out2)                                           #124 * 124 * 64
        #print('out3:{}'.format(out3.shape))
        out4 = F.relu(self.conv3(out3))                                  #122 * 122 * 128
        #print('out4:{}'.format(out4.shape))
        out5 = F.relu(self.conv4(out4))                                  #120 * 120 * 128
        #print('out5:{}'.format(out5.shape))
        out6 = self.pool(out5)                                           #60 * 60 * 128
        #print('out6:{}'.format(out6.shape))
        out7 = F.relu(self.conv5(out6))                                  #58 * 58 * 256
        #print('out7:{}'.format(out7.shape))
        out8 = F.relu(self.conv6(out7))                                  #56 * 56 * 256
        #print('out8:{}'.format(out8.shape))
        out9 = self.pool(out8)                                           #28 * 28 * 256
        #print('out9:{}'.format(out9.shape))
        out10 = F.relu(self.conv7(out9))                                 #26 * 26 * 512
        #print('out10:{}'.format(out10.shape))
        out11 = F.relu(self.conv8(out10))                                #24 * 24 * 512
        #print('out11:{}'.format(out11.shape))
        out12 = self.pool(out11)                                         #12 * 12 * 512
        #print('out12:{}'.format(out12.shape))
        out13 = F.relu(self.conv9(out12))                                #10 * 10 * 1024
        #print('out13:{}'.format(out13.shape))
        out14 = F.relu(self.conv10(out13))                               #8 * 8 * 1024
        #print('out14:{}'.format(out14.shape))

        out15 = self.upsample(out14)                                     #16 * 16 * 1024  
        #print('out15:{}'.format(out15.shape))
        out16 = self.upConv1(out15)                                      #16 * 16 * 512
        #print('out16:{}'.format(out16.shape))
        out16_bypass = out11[:,:,4:20,4:20]
        #print('out16:{}'.format(out16.shape))
        out17 = torch.cat((out16, out16_bypass), 1)                      #16 * 16 * 1024
        #print('out17:{}'.format(out17.shape))
        out18 = F.relu(self.deConv1(out17))                              #14 * 14 * 512
        #print('out18:{}'.format(out18.shape))
        out19 = F.relu(self.conv8(out18))                                #12 * 12 * 512
        #print('out19:{}'.format(out19.shape))

        out20 = self.upsample(out19)                                     #24 * 24 * 512
        #print('out20:{}'.format(out20.shape))
        out21 = self.upConv2(out20)                                      #24 * 24 * 256
        #print('out21:{}'.format(out21.shape))
        out21_bypass = out8[:, :, 16:40, 16:40]                          #24 * 24 * 256
        #print('out21_bypass:{}'.format(out21_bypass.shape))
        out22 = torch.cat((out21, out21_bypass), 1)                      #24 * 24 * 512 
        #print('out22:{}'.format(out22.shape))
        out23 = F.relu(self.deConv2(out22))                              #22 * 22 * 256
        #print('out23:{}'.format(out23.shape))
        out24 = F.relu(self.conv6(out23))                                #20 * 20 * 256
        #print('out24:{}'.format(out24.shape))

        out25 = self.upsample(out24)                                     #40 * 40 * 256
        #print('out25:{}'.format(out25.shape))
        out26 = self.upConv3(out25)                                      #40 * 40 * 128
        #print('out26:{}'.format(out26.shape))
        out26_bypass = out5[:, :, 40:80, 40:80]                          #40 * 40 * 128
        #print('out26_bypass:{}'.format(out26_bypass.shape))
        out27 = torch.cat((out26, out26_bypass), 1)                      #40 * 40 * 256 
        #print('out27:{}'.format(out27.shape))
        out28 = F.relu(self.deConv3(out27))                              #38 * 38 * 128
        #print('out28:{}'.format(out28.shape))
        out29 = F.relu(self.conv4(out28))                                #36 * 36 * 128
        #print('out29:{}'.format(out29.shape))

        out30 = self.upsample(out29)                                     #72 * 72 * 128
        #print('out30:{}'.format(out30.shape))
        out31 = self.upConv4(out30)                                      #72 * 72 * 64
        #print('out31:{}'.format(out31.shape))
        out31_bypass = out2[:, :, 88:160, 88:160]                        #72 * 72  * 64
        #print('out31_bypass:{}'.format(out31_bypass.shape))
        out32 = torch.cat((out31, out31_bypass), 1)                      #72 * 72 * 128
        #print('out32:{}'.format(out32.shape))
        out33 = self.deConv4(out32)                                      #70 * 70 * 64
        #print('out33:{}'.format(out33.shape))
        out34 = self.conv2(out33)                                        #68 * 68 * 64
        #print('out34:{}'.format(out34.shape))
        out35 = self.deConv5(out34)                                      #68 * 68 * 1
        #print('out35:{}'.format(out35.shape))
        return out35

    def _initialize_weights(self):
        self.apply(init_weights)

