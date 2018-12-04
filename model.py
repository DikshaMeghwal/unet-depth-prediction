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
        super(UNet, self).__init__()                                   #256 * 256 * 3 - input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)                   #256 * 256 * 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)                  #256 * 256 * 64
                                                                       #after maxpool 128 * 128 * 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)                 #128 * 128 * 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)                #128 * 128 * 128
                                                                       #after maxpool 64 * 64 * 128
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)                #64 * 64 * 256
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)                #64 * 64 * 64
                                                                       #after maxpool 32 * 32 * 256
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)                #32 * 32 * 512
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)                #32 * 32 * 512
                                                                       #after maxpool 16 * 16 * 512
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)               #16 * 16 * 1024
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)             #16 * 16 * 1024

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')      #32 * 32 * 1024  
        self.upConv1 = nn.Conv2d(1024, 512, kernel_size=1)               #32 * 32 * 512
                                                                         #after concatenation with 32 * 32 * 512
                                                                         #32 * 32 * 1024
        self.deConv1 = nn.Conv2d(1024, 512, kernel_size = 3, padding=1)             #32 * 32 * 512
        #self.conv8                                                      #32 * 32 * 512

        #upsample 1                                                      #64 * 64 * 512
        self.upConv2 = nn.Conv2d(512, 256, kernel_size=1)                #64 * 64 * 256
                                                                         #after concatenation with 64 * 64 * 256
                                                                         #64 * 64 * 512 
        self.deConv2 = nn.Conv2d(512, 256, kernel_size = 3, padding=1)              #64 * 64 * 256
        #self.conv6                                                      #64 * 64 * 256

        #upsample 1                                                      #128 * 128 * 256
        self.upConv3 = nn.Conv2d(256, 128, kernel_size=1)                #128 * 128 * 128
                                                                         #after concatenation with 128 * 128 * 128
                                                                         #128 * 128 * 256 
        self.deConv3 = nn.Conv2d(256, 128, kernel_size = 3, padding=1)              #128 * 128 * 128
        #self.conv4                                                      #128 * 128 * 128

        #upsample 1                                                      #256 * 256 * 128
        self.upConv4 = nn.Conv2d(128, 64, kernel_size=1)                 #256 * 256 * 64
                                                                         #after concatenation with crop of 256 * 256 * 64
                                                                         #256 * 256 * 128
        self.deConv4 = nn.Conv2d(128, 64, kernel_size = 3, padding=1)               #256 * 256 * 64
        #self.conv2                                                      #256 * 256 * 64
        self.deConv5 = nn.Conv2d(64, 1, kernel_size = 1)                 #256 * 256 * 1

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
                                                                         
                                                                         
        out1 = F.relu(self.conv1(x))                                     
        #print('out1:{}'.format(out1.shape))
        out2 = F.relu(self.conv2(out1))                                  
        #print('out2:{}'.format(out2.shape))
        out3 = self.pool(out2)                                           
        #print('out3:{}'.format(out3.shape))
        out4 = F.relu(self.conv3(out3))                                  
        #print('out4:{}'.format(out4.shape))
        out5 = F.relu(self.conv4(out4))                                  
        #print('out5:{}'.format(out5.shape))
        out6 = self.pool(out5)                                           
        #print('out6:{}'.format(out6.shape))
        out7 = F.relu(self.conv5(out6))                                  
        #print('out7:{}'.format(out7.shape))
        out8 = F.relu(self.conv6(out7))                                  
        #print('out8:{}'.format(out8.shape))
        out9 = self.pool(out8)                                           
        #print('out9:{}'.format(out9.shape))
        out10 = F.relu(self.conv7(out9))                                 
        #print('out10:{}'.format(out10.shape))
        out11 = F.relu(self.conv8(out10))                                
        #print('out11:{}'.format(out11.shape))
        out12 = self.pool(out11)                                         
        #print('out12:{}'.format(out12.shape))
        out13 = F.relu(self.conv9(out12))                                
        #print('out13:{}'.format(out13.shape))
        out14 = F.relu(self.conv10(out13))                               
        #print('out14:{}'.format(out14.shape))

        out15 = self.upsample(out14)                                     
        #print('out15:{}'.format(out15.shape))
        out16 = self.upConv1(out15)                                      
        #print('out16:{}'.format(out16.shape))
        out17 = torch.cat((out16, out11), 1)                             
        #print('out17:{}'.format(out17.shape))
        out18 = F.relu(self.deConv1(out17))                              
        #print('out18:{}'.format(out18.shape))
        out19 = F.relu(self.conv8(out18))                                
        #print('out19:{}'.format(out19.shape))

        out20 = self.upsample(out19)                                     
        #print('out20:{}'.format(out20.shape))
        out21 = self.upConv2(out20)                                      
        #print('out21:{}'.format(out21.shape))
        out22 = torch.cat((out21, out8), 1)                              
        #print('out22:{}'.format(out22.shape))
        out23 = F.relu(self.deConv2(out22))                              
        #print('out23:{}'.format(out23.shape))
        out24 = F.relu(self.conv6(out23))                                
        #print('out24:{}'.format(out24.shape))

        out25 = self.upsample(out24)                                     
        #print('out25:{}'.format(out25.shape))
        out26 = self.upConv3(out25)                                      
        #print('out26:{}'.format(out26.shape))
        out27 = torch.cat((out26, out5), 1)                              
        #print('out27:{}'.format(out27.shape))
        out28 = F.relu(self.deConv3(out27))                              
        #print('out28:{}'.format(out28.shape))
        out29 = F.relu(self.conv4(out28))                                
        #print('out29:{}'.format(out29.shape))

        out30 = self.upsample(out29)                                     
        #print('out30:{}'.format(out30.shape))
        out31 = self.upConv4(out30)                                      
        #print('out31:{}'.format(out31.shape))
        out32 = torch.cat((out31, out2), 1)                              
        #print('out32:{}'.format(out32.shape))
        out33 = self.deConv4(out32)                                      
        #print('out33:{}'.format(out33.shape))
        out34 = self.conv2(out33)                                        
        #print('out34:{}'.format(out34.shape))
        out35 = self.deConv5(out34)                                      
        #print('out35:{}'.format(out35.shape))
        return out35

    def _initialize_weights(self):
        self.apply(init_weights)
