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
        super(UNet, self).__init__()                                #572 * 572 * 3 - input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)                   #570 * 570 * 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)                  #568 * 568 * 64
                                                                         #after maxpool 284 * 284 * 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)                 #282 * 282 * 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)                #280 * 280 * 128
                                                                         #after maxpool 140 * 140 * 128
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)                #138 * 138 * 256
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)                #136 * 136 * 256
                                                                         #after maxpool 68 * 68 * 256
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3)                #66 * 66 * 512
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3)                #64 * 64 * 512
                                                                         #after maxpool 32 * 32 * 512
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3)               #30 * 30 * 1024
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3)             #28 * 28 * 1024

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')      #56 * 56 * 1024  
        self.upConv1 = nn.Conv2d(1024, 512, kernel_size=1)               #56 * 56 * 512
                                                                         #after concatenation with crop of 56 * 56 * 512
                                                                         #56 * 56 * 1024
        self.deConv1 = nn.Conv2d(1024, 512, kernel_size = 3)             #54 * 54 * 512
        #self.conv8                                                      #52 * 52 * 512

        #upsample 1                                                      #104 * 104 * 512
        self.upConv2 = nn.Conv2d(512, 256, kernel_size=1)                #104 * 104 * 256
                                                                         #after concatenation with crop of 104 * 104 * 256
                                                                         #104 * 104 * 512 
        self.deConv2 = nn.Conv2d(512, 256, kernel_size = 3)              #102 * 102 * 256
        #self.conv6                                                      #100 * 100 * 256

        #upsample 1                                                      #200 * 200 * 256
        self.upConv3 = nn.Conv2d(256, 128, kernel_size=1)                #200 * 200 * 128
                                                                         #after concatenation with crop of 200 * 200 * 128
                                                                         #200 * 200 * 256 
        self.deConv3 = nn.Conv2d(256, 128, kernel_size = 3)              #198 * 198 * 128
        #self.conv4                                                      #196 * 196 * 128

        #upsample 1                                                      #392 * 392 * 128
        self.upConv4 = nn.Conv2d(128, 64, kernel_size=1)                 #392 * 392 * 64
                                                                         #after concatenation with crop of 392 * 392 * 64
                                                                         #392 * 392 * 128
        self.deConv4 = nn.Conv2d(128, 64, kernel_size = 3)               #390 * 390 * 64
        #self.conv2                                                      #388 * 388 * 64
        self.deConv5 = nn.Conv2d(64, 1, kernel_size = 1)                 #388 * 388 * 1

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
                                                                         #H * W * c
                                                                         #572 * 572 * 3
        out1 = F.relu(self.conv1(x))                                     #570 * 570 * 64
        #print('out1:{}'.format(out1.shape))
        out2 = F.relu(self.conv2(out1))                                  #568 * 568 * 64
        #print('out2:{}'.format(out2.shape))
        out3 = self.pool(out2)                                           #284 * 284 * 64
        #print('out3:{}'.format(out3.shape))
        out4 = F.relu(self.conv3(out3))                                  #282 * 282 * 128
        #print('out4:{}'.format(out4.shape))
        out5 = F.relu(self.conv4(out4))                                  #280 * 280 * 128
        #print('out5:{}'.format(out5.shape))
        out6 = self.pool(out5)                                           #140 * 140 * 128
        #print('out6:{}'.format(out6.shape))
        out7 = F.relu(self.conv5(out6))                                  #138 * 138 * 256
        #print('out7:{}'.format(out7.shape))
        out8 = F.relu(self.conv6(out7))                                  #136 * 136 * 256
        #print('out8:{}'.format(out8.shape))
        out9 = self.pool(out8)                                           #68 * 68 * 256
        #print('out9:{}'.format(out9.shape))
        out10 = F.relu(self.conv7(out9))                                 #66 * 66 * 512
        #print('out10:{}'.format(out10.shape))
        out11 = F.relu(self.conv8(out10))                                #64 * 64 * 512
        #print('out11:{}'.format(out11.shape))
        out12 = self.pool(out11)                                         #32 * 32 * 512
        #print('out12:{}'.format(out12.shape))
        out13 = F.relu(self.conv9(out12))                                #30 * 30 * 1024
        #print('out13:{}'.format(out13.shape))
        out14 = F.relu(self.conv10(out13))                               #28 * 28 * 1024
        #print('out14:{}'.format(out14.shape))

        out15 = self.upsample(out14)                                     #56 * 56 * 1024  
        #print('out15:{}'.format(out15.shape))
        out16 = self.upConv1(out15)                                      #56 * 56 * 512
        #print('out16:{}'.format(out16.shape))
        out16_bypass = out11[:,:,4:60,4:60]
        #print('out16:{}'.format(out16.shape))
        out17 = torch.cat((out16, out16_bypass), 1)                      #56 * 56 * 1024
        #print('out17:{}'.format(out17.shape))
        out18 = F.relu(self.deConv1(out17))                              #54 * 54 * 512
        #print('out18:{}'.format(out18.shape))
        out19 = F.relu(self.conv8(out18))                                #52 * 52 * 512
        #print('out19:{}'.format(out19.shape))

        out20 = self.upsample(out19)                                     #104 * 104 * 512
        #print('out20:{}'.format(out20.shape))
        out21 = self.upConv2(out20)                                      #104 * 104 * 256
        #print('out21:{}'.format(out21.shape))
        out21_bypass = out8[:, :, 16:120, 16:120]                        #104 * 104 * 256
        #print('out21_bypass:{}'.format(out21_bypass.shape))
        out22 = torch.cat((out21, out21_bypass), 1)                      #104 * 104 * 512 
        #print('out22:{}'.format(out22.shape))
        out23 = F.relu(self.deConv2(out22))                              #102 * 102 * 256
        #print('out23:{}'.format(out23.shape))
        out24 = F.relu(self.conv6(out23))                                #100 * 100 * 256
        #print('out24:{}'.format(out24.shape))

        out25 = self.upsample(out24)                                     #200 * 200 * 256
        #print('out25:{}'.format(out25.shape))
        out26 = self.upConv3(out25)                                      #200 * 200 * 128
        #print('out26:{}'.format(out26.shape))
        out26_bypass = out5[:, :, 40:240, 40:240]                        #200 * 200 * 128
        #print('out26_bypass:{}'.format(out26_bypass.shape))
        out27 = torch.cat((out26, out26_bypass), 1)                      #200 * 200 * 256 
        #print('out27:{}'.format(out27.shape))
        out28 = F.relu(self.deConv3(out27))                              #198 * 198 * 128
        #print('out28:{}'.format(out28.shape))
        out29 = F.relu(self.conv4(out28))                                #196 * 196 * 128
        #print('out29:{}'.format(out29.shape))

        out30 = self.upsample(out29)                                     #392 * 392 * 128
        #print('out30:{}'.format(out30.shape))
        out31 = self.upConv4(out30)                                      #392 * 392 * 64
        #print('out31:{}'.format(out31.shape))
        out31_bypass = out2[:, :, 88:480, 88:480]                        #392 * 392 * 64
        #print('out31_bypass:{}'.format(out31_bypass.shape))
        out32 = torch.cat((out31, out31_bypass), 1)                      #392 * 392 * 128
        #print('out32:{}'.format(out32.shape))
        out33 = self.deConv4(out32)                                      #390 * 390 * 64
        #print('out33:{}'.format(out33.shape))
        out34 = self.conv2(out33)                                        #388 * 388 * 64
        #print('out34:{}'.format(out34.shape))
        out35 = self.deConv5(out34)                                      #388 * 388 * 1
        #print('out35:{}'.format(out35.shape))
        return out35

    def _initialize_weights(self):
        self.apply(init_weights)
