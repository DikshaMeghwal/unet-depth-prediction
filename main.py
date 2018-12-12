from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from torch.autograd import Variable
from logger import Logger
import pdb
import os
import re
import numpy as np
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch depth map prediction example')
parser.add_argument('model_folder', type=str, default='trial', metavar='F',
                     help='In which folder do you want to save the model')
parser.add_argument('--data', type=str, default='data', metavar='D',
                     help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type = int, default = 32, metavar = 'N',
                     help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default = 10, metavar='N',
                      help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                     help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                     help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                     help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                     help='how many batches to wait before logging training status')
parser.add_argument('--suffix', type=str, default='', metavar='D',
                     help='suffix for the filename of models and output files')
args = parser.parse_args()

from data import NYUDataset, rgb_data_transforms, depth_data_transforms, input_for_plot_transforms, output_height, output_width

train_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat', 
                                                       'training', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 5)

val_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'validation', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 5)


from tiny_unet import UNet
model = UNet()
model.cuda()

def custom_loss_function(output, target):
    di = target - output
    n = (output_height * output_width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.sum()

loss_function = custom_loss_function
#optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=0.0001)
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
dtype=torch.cuda.FloatTensor
logger = Logger('./logs/' + args.model_folder)

def plot_n_save_fig(epoch, plot_input, output, actual_output):
    F = plt.figure(1, (30, 60))
    F.subplots_adjust(left=0.05, right=0.95)
    plot_grid(F, plot_input, output, actual_output, 1)
    plt.savefig("plots/" + args.model_folder + "_" + str(epoch) + ".jpg")
    plt.show()

def plot_grid(fig, plot_input, output, actual_output, row_no):
    grid = ImageGrid(fig, 141, nrows_ncols=(row_no, 4), axes_pad=0.05, label_mode="1")
    for i in range(row_no):
        for j in range(3):
            if(j == 0):
                grid[i*4+j].imshow(np.transpose(plot_input[i], (1, 2, 0)), interpolation="nearest")
            if(j == 1):
                grid[i*4+j].imshow(np.transpose(output[i][0].detach().cpu().numpy(), (0, 1)), interpolation="nearest")
            if(j == 2):
                grid[i*4+j].imshow(np.transpose(actual_output[i][0].detach().cpu().numpy(), (0, 1)), interpolation="nearest")

def train_Unet(epoch):
    model.train()
    for batch_idx, image in enumerate(train_loader):
#        start = time.time()
        x = image['image'].cuda()
        y = image['depth'].cuda()

        optimizer.zero_grad()
        y_hat = model(x.type(dtype))
        loss = custom_loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

        if ((epoch-1) % 50 == 0) and batch_idx == 0:
            to_print = "Training epoch[{}/{}] Loss: {:.3f}".format(epoch, args.epochs, loss.item()) 
            plot_n_save_fig(batch_idx, x, y_hat, y)
            print(to_print)
#        end = time.time()
#        print("training time for single batch for epoch {} is".format(epoch) + str(end - start))
#        if batch_idx == 0: break

def validate_Unet():
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for idx, image in enumerate(val_loader):
            x = image['image'].cuda()
            y = image['depth'].cuda()

            y_hat = model(x.type(dtype))
            loss = custom_loss_function(y_hat, y)
            validation_loss += loss
#            if idx == 5: break
        validation_loss /= idx
#        print("validation time for single batch for epoch {} is".format(epoch) + str(end - start))
 
folder_name = "models/" + args.model_folder
if not os.path.exists(folder_name): os.mkdir(folder_name)

for epoch in range(1, args.epochs + 1):
    print("********* Training the Unet Model **************")
    print("epoch:{}".format(epoch)) 
#    start = time.time()
    train_Unet(epoch)
#    end = time.time()
#    print("training time for epoch {} is".format(epoch) + str(end - start))
#    start = time.time()
#    end = time.time()
#    print("validation time for epoch {} is".format(epoch) + str(end - start))
    if epoch % 50 == 0:
        validate_Unet()
        model_file = folder_name + "/model_" + str(epoch) + ".pth"    
        torch.save(model.state_dict(), model_file)

