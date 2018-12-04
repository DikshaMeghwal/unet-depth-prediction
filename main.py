from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from torch.autograd import Variable
from logger import Logger
import pdb
import os
import re

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
from data import initialize_data, rgb_data_transforms, depth_data_transforms, output_height, output_width
initialize_data(args.data) # extracts the zip files, makes a validation set

train_rgb_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/train_images/rgb/', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=True, num_workers=1)
train_depth_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/train_images/depth/', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=True, num_workers=1)
val_rgb_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/val_images/rgb/', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
val_depth_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/val_images/depth/', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)

from model import UNet
model = UNet()
model.cuda()
# loss_function = nn.MSELoss()
loss_function = F.smooth_l1_loss
# optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=0.0001)
optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum=0.99)
dtype=torch.cuda.FloatTensor
logger = Logger('./logs/' + args.model_folder)

def display_images(images):
#     print("printing image of size")
#     print(images.shape)
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

def train_Unet(epoch):
    model.train()
    for batch_idx, (rgb, depth) in enumerate(zip(train_rgb_loader, train_depth_loader)):
        rgb, depth = Variable(rgb[0].cuda()), Variable(depth[0].cuda())
        optimizer.zero_grad()
        output = model(rgb.type(dtype))
#        print('printing output shape')
#        print(output.shape)
#             print("printing weights")
#             output_data = format_data_for_display(output[3])
#             display_images(output_data)
#             output_data = format_data_for_display(output[20])
#             display_images(output_data)
        target = depth[:,0,:,:].view(list(depth.shape)[0], 1, output_height, output_width)
        # loss = loss_function(output, depth[:,0,:,:].view(batch_size, 1, output_height, output_width))
        loss = loss_function(output, target)
#        if batch_idx == 0:
#             print('input:')
#             print(rgb.data)
#             display_images(rgb.data)
#             print('output:')
#             print(output)
#             format_output = format_data_for_display(output)
#             display_images(format_output)
#             print('target:')
#             print(target)
#             format_target = format_data_for_display(target)
#             display_images(format_target)
#             print(loss)
        loss.backward()
        optimizer.step()
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
#         if batch_idx == 2: break

def validate_Unet():
    print('validating unet')
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch_idx,(rgb, depth) in enumerate(zip(val_rgb_loader, val_depth_loader)):
            rgb, depth = Variable(rgb[0].cuda()), Variable(depth[0].cuda())
            output = model(rgb.type(dtype))
            target = depth[:,0,:,:].view(list(depth.shape)[0], 1, output_height, output_width)
            validation_loss += loss_function(output, target)
#           if batch_idx == 2: break
        validation_loss /= batch_idx
        logger.scalar_summary("validation loss", validation_loss, epoch)
        print('\nValidation set: Average loss: {:.4f} \n'.format(validation_loss))

folder_name = "models/" + args.model_folder
if not os.path.exists(folder_name): os.mkdir(folder_name)

for epoch in range(1, args.epochs + 1):
    print("********* Training the Unet Model **************")
    train_Unet(epoch)
    model_file = folder_name + "/" + 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    validate_Unet()

