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

#data = 'data'
#batch_size = 8
#epochs = 20
#lr = 0.0001
#momentum = 0.5
#seed = 1
#log_interval = 10
#suffix = ''
#model_folder = 'local-unet'

#torch.manual_seed(seed)

### Data Initialization and Loading
# from data import initialize_data, rgb_data_transforms, depth_data_transforms, output_height, output_width
initialize_data(args.data) # extracts the zip files, makes a validation set

from data import NYUDataset, rgb_data_transforms, depth_data_transforms, input_for_plot_transforms

# train_rgb_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/train_images/rgb/', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
# train_depth_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/train_images/depth/', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
# val_rgb_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/val_images/rgb/', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
# val_depth_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/val_images/depth/', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)

train_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat', 
                                                       'training', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = True, num_workers = 0)

val_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'validation', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 0)


from model import UNet
model = UNet()
model.cuda()

def rel_error(output, target):
    target = target + 0.000001
    target = log10(target)
    output = output + 0.000001
    output = log10(output)
    return F.mse_loss(output, target)
    #diff = (output-target)/target
    #diff = torch.abs(diff)
    #return diff.mean()

loss_function = F.mse_loss
#loss_function = F.smooth_l1_loss
#loss_function = rel_error
optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=0.0001)
#optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.99)
#optimizer = optim.Adamax(model.parameters())
dtype=torch.cuda.FloatTensor
logger = Logger('./logs/' + args.model_folder)

def display_images(images):
    grid = utils.make_grid(images)
    plt.imshow(grid.cpu().detach().numpy().transpose((1, 2, 0)))
    plt.show();

def format_data_for_display(tensor):
    maxVal = tensor.max()
    minVal = abs(tensor.min())
    maxVal = max(maxVal,minVal)
    output_data = tensor / maxVal
    output_data = output_data / 2
    output_data = output_data + 0.5
    return output_data

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
    for batch_idx, data in enumerate(train_loader):
        rgb, depth = data['image'].cuda(), data['depth'].cuda()
        optimizer.zero_grad()
        output = model(rgb.type(dtype))
        target = depth[:,0,:,:].view(list(depth.shape)[0], 1, output_height, output_width)
        #print("target")
        #print(target)
        #print("output")
        #print(output)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        F = plt.figure(1, (30, 60))
        F.subplots_adjust(left=0.05, right=0.95)
        plot_grid(F, rgb, target, output, args.batch_size)
        plt.savefig("plots/train_" + args.model_folder + "_" + str(epoch) + "_" + str(batch_idx) + ".jpg")
        plt.show()

        if batch_idx % args.log_interval == 0:
            training_tag = "training loss epoch:" + str(epoch)
            logger.scalar_summary(training_tag, loss.item(), batch_idx)

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/') + ":" + str(epoch)
                #logger.histo_summary(tag, value.data.cpu().numpy(), batch_idx)
                #logger.histo_summary(tag + '/grad', value.grad.data.cpu().detach().numpy(), batch_idx)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(rgb), len(train_rgb_loader.dataset),
                100. * batch_idx / len(train_rgb_loader), loss.item()))
#         batch_idx = batch_idx + 1
        if batch_idx == 0: break

def validate_Unet():
    print('validating unet')
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            rgb, depth = data['image'].cuda(), data['depth'].cuda()
            output = model(rgb.type(dtype))
            target = depth[:,0,:,:].view(list(depth.shape)[0], 1, output_height, output_width)
            validation_loss += rel_error(output, target)
#           if batch_idx == 2: break
            rel_loss = rel_error(output, target)
            rms_loss = F.mse_loss(output, target)
        validation_loss /= batch_idx
        rel_loss /= batch_idx
        rms_loss /= batch_idx
        logger.scalar_summary("validation loss", validation_loss, epoch)
        print('\nValidation set: Average loss: {:.6f} {:.6f} {:.6f}\n'.format(validation_loss, rel_loss, rms_loss))

folder_name = "models/" + args.model_folder
if not os.path.exists(folder_name): os.mkdir(folder_name)

for epoch in range(1, args.epochs + 1):
    print("********* Training the Unet Model **************")
    train_Unet(epoch)
    model_file = folder_name + "/" + 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
#    validate_Unet()

