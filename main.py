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
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                     help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                     help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                     help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
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
                                            shuffle = True, num_workers = 5)

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
    return loss.mean()

# loss_function = custom_loss_function
# optimizer = optim.Adam(model.parameters(), amsgrad=True, lr=0.001)

# Paper values for SGD
# coarse_optimizer = optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.001},{'params': coarse_model.conv2.parameters(), 'lr': 0.001},{'params': coarse_model.conv3.parameters(), 'lr': 0.001},{'params': coarse_model.conv4.parameters(), 'lr': 0.001},{'params': coarse_model.conv5.parameters(), 'lr': 0.001},{'params': coarse_model.fc1.parameters(), 'lr': 0.1},{'params': coarse_model.fc2.parameters(), 'lr': 0.1}], lr = 0.001, momentum = 0.9)
# fine_optimizer = optim.SGD([{'params': fine_model.conv1.parameters(), 'lr': 0.001},{'params': fine_model.conv2.parameters(), 'lr': 0.01},{'params': fine_model.conv3.parameters(), 'lr': 0.001}], lr = 0.001, momentum = 0.9)

# Changed values
# coarse_optimizer = optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.01},{'params': coarse_model.conv2.parameters(), 'lr': 0.01},{'params': coarse_model.conv3.parameters(), 'lr': 0.01},{'params': coarse_model.conv4.parameters(), 'lr': 0.01},{'params': coarse_model.conv5.parameters(), 'lr': 0.01},{'params': coarse_model.fc1.parameters(), 'lr': 0.1},{'params': coarse_model.fc2.parameters(), 'lr': 0.1}], lr = 0.01, momentum = 0.9)
# fine_optimizer = optim.SGD(fine_model.parameters(), lr=args.lr, momentum=args.momentum)
# fine modified but default fine work more.
#fine_optimizer = optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.01},{'params': coarse_model.conv2.parameters(), 'lr': 0.1},{'params': coarse_model.conv3.parameters(), 'lr': 0.01}], lr = 0.01, momentum = 0.9)

# default SGD optimiser - don't work
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)
# fine_optimizer = optim.SGD(fine_model.parameters(), lr=args.lr, momentum=args.momentum)

# coarse_optimizer = optim.Adadelta(coarse_model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
# fine_optimizer = optim.Adadelta(fine_model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

# coarse_optimizer = optim.Adagrad(coarse_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
# fine_optimizer = optim.Adagrad(fine_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

# coarse_optimizer = optim.Adam(coarse_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# fine_optimizer = optim.Adam(fine_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# coarse_optimizer = optim.Adamax(coarse_model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# fine_optimizer = optim.Adamax(fine_model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# coarse_optimizer = optim.ASGD(coarse_model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
# fine_optimizer = optim.ASGD(fine_model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

dtype=torch.cuda.FloatTensor
si_logger = Logger('./logs/' + args.model_folder + '/scale_invariant_validation_loss')
t1_logger = Logger('./logs/' + args.model_folder + '/threshold_lt_1.25')
t2_logger = Logger('./logs/' + args.model_folder + '/threshold_lt_1.25sq')
t3_logger = Logger('./logs/' + args.model_folder + '/threshold_lt_1.25cb')
rmse_logger = Logger('./logs/' + args.model_folder + '/rmse_linear')
rmse_log_logger = Logger('./logs/' + args.model_folder + '/rmse_log')
abs_rel_diff_logger = Logger('./logs/' + args.model_folder + '/abs_rel_diff')
abs_rel_diff_sq_logger = Logger('./logs/' + args.model_folder + '/abs_rel_diff_sq')

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

# All Error Function
def threeshold_percentage(output, target, threeshold_val):
    d1 = torch.exp(output)/torch.exp(target)
    d2 = torch.exp(target)/torch.exp(output)
    max_d1_d2 = torch.max(d1,d2)
    zero = torch.zeros(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    one = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    bit_mat = torch.where(max_d1_d2.cpu() < threeshold_val, one, zero)
    count_mat = torch.sum(bit_mat, (1,2,3))
    threeshold_mat = count_mat/(output.shape[2] * output.shape[3])
    return threeshold_mat.mean()

def rmse_linear(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    diff = actual_output - actual_target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1,2,3))/(output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return rmse.mean()

def rmse_log(output, target):
    diff = output - target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1,2,3))/(output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return mse.mean()

def abs_relative_difference(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    abs_relative_diff = torch.abs(actual_output - actual_target)/actual_target
    abs_relative_diff = torch.sum(abs_relative_diff, (1,2,3))/(output.shape[2] * output.shape[3])
    return abs_relative_diff.mean()

def squared_relative_difference(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    square_relative_diff = torch.pow(torch.abs(actual_output - actual_target), 2)/actual_target
    square_relative_diff = torch.sum(square_relative_diff, (1,2,3))/(output.shape[2] * output.shape[3])
    return square_relative_diff.mean()
#############################################

def train_Unet(epoch):
    model.train()
    train_coarse_loss = 0
    for batch_idx, image in enumerate(train_loader):
#        start = time.time()
        # pdb.set_trace()
        x = image['image'].cuda()
        y = image['depth'].cuda()

        optimizer.zero_grad()
        y_hat = model(x.type(dtype))
        loss = custom_loss_function(y_hat, y)
        loss.backward()
        optimizer.step()
        train_coarse_loss += loss.item()
        if epoch % args.log_interval==0:
            training_tag = "training loss epoch:" + str(epoch)
            #logger.scalar_summary(training_tag, loss.item(), batch_idx)
    train_coarse_loss /= (batch_idx + 1)
    return train_coarse_loss
print("Epochs:     Train_loss  Val_loss    Delta_1     Delta_2     Delta_3    rmse_lin    rmse_log    abs_rel.  square_relative")
print("Paper Val:                          (0.618)     (0.891)     (0.969)     (0.871)     (0.283)     (0.228)     (0.223)")
def validate_Unet(epoch, training_loss):
    model.eval()
    validation_loss = 0
    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0
    with torch.no_grad():
        for idx, image in enumerate(val_loader):
            # pdb.set_trace()
            x = image['image'].cuda()
            y = image['depth'].cuda()

            y_hat = model(x.type(dtype))
            loss = custom_loss_function(y_hat, y)
            validation_loss += loss
            # all error functions
            delta1_accuracy += threeshold_percentage(y_hat, y, 1.25)
            delta2_accuracy += threeshold_percentage(y_hat, y, 1.25*1.25)
            delta3_accuracy += threeshold_percentage(y_hat, y, 1.25*1.25*1.25)
            rmse_linear_loss += rmse_linear(y_hat, y)
            rmse_log_loss += rmse_log(y_hat, y)
            abs_relative_difference_loss += abs_relative_difference(y_hat, y)
            squared_relative_difference_loss += squared_relative_difference(y_hat, y)
        validation_loss /= (idx + 1)
        delta1_accuracy /= (idx + 1)
        delta2_accuracy /= (idx + 1)
        delta3_accuracy /= (idx + 1)
        rmse_linear_loss /= (idx + 1)
        rmse_log_loss /= (idx + 1)
        abs_relative_difference_loss /= (idx + 1)
        squared_relative_difference_loss /= (idx + 1)
        print('Epoch: {}    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.
            format(epoch, training_loss, validation_loss, delta1_accuracy, delta2_accuracy, delta3_accuracy, rmse_linear_loss, rmse_log_loss, 
            abs_relative_difference_loss, squared_relative_difference_loss))
        si_logger.scalar_summary("validation loss", validation_loss, epoch)
        t1_logger.scalar_summary("validation loss", delta1_accuracy, epoch)
        t2_logger.scalar_summary("validation loss", delta2_accuracy, epoch)
        t3_logger.scalar_summary("validation loss", delta3_accuracy, epoch)
        rmse_logger.scalar_summary("validation loss", rmse_linear_loss, epoch)
        rmse_log_logger.scalar_summary("validation loss", rmse_log_loss, epoch)
        abs_rel_diff_logger.scalar_summary("validation loss", abs_relative_difference_loss, epoch)
        abs_rel_diff_sq_logger.scalar_summary("validation loss", squared_relative_difference_loss, epoch)

folder_name = "models/" + args.model_folder
if not os.path.exists(folder_name): os.mkdir(folder_name)

print("********* Training the Unet Model **************")
for epoch in range(1, args.epochs + 1):
    training_loss = train_Unet(epoch)
    if epoch % 1 == 0:
        validate_Unet(epoch, training_loss)
    if epoch % 1 == 0:
        model_file = folder_name + "/model_" + str(epoch) + ".pth"    
        torch.save(model.state_dict(), model_file)

