import matplotlib
import argparse
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from tiny_unet import UNet
import pdb
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch depth prediction evaluation script')
parser.add_argument('model_folder', type=str, metavar='F',
                    help='In which folder have you saved the model')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model_no', type=int, default = 1900, metavar='N',
                    help='Which model no to evaluate (default: 1(first model))')
parser.add_argument('--batch-size', type = int, default = 8, metavar = 'N',
                    help='input batch size for training (default: 8)')

args = parser.parse_args()

from data import output_height, output_width

state_dict = torch.load("models/" + args.model_folder + "/model_" + str(args.model_no) + ".pth")

model = UNet()
model.cuda()

dtype=torch.cuda.FloatTensor

model.load_state_dict(state_dict)
model.eval()

# from data import rgb_data_transforms, depth_data_transforms, input_for_plot_transforms
# test_rgb_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/test_images/rgb/', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
# test_depth_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/test_images/depth/', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
# input_for_plot_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/test_images/rgb/', transform = input_for_plot_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)

from data import NYUDataset, input_for_plot_transforms, rgb_data_transforms, depth_data_transforms

test_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'test', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 50)

input_for_plot_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat', 
                                                                'test', 
                                                                rgb_transform = input_for_plot_transforms, 
                                                                depth_transform = depth_data_transforms), 
                                                    batch_size = args.batch_size, 
                                                    shuffle = False, num_workers = 50)


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

for batch_idx, data in enumerate(test_loader):
    rgb, depth = torch.tensor(data['image'], requires_grad = False).cuda(), torch.tensor(data['depth'], requires_grad = False).cuda()
    # plot_input, actual_output = torch.tensor(plot_data['image'], requires_grad = False), torch.tensor(plot_data['depth'], requires_grad = False)
    plot_input = rgb.type(dtype)
    actual_output = depth.type(dtype)

    print('evaluating batch:' + str(batch_idx))
    output = model(rgb)
    depth_dim = list(depth.size())
    # actual_output = depth[:,0,:,:].view(depth_dim[0], 1, output_height, output_width)
    F = plt.figure(1, (30, 60))
    F.subplots_adjust(left=0.05, right=0.95)
    plot_grid(F, plot_input, output, actual_output, depth_dim[0])
    plt.savefig("plots/" + args.model_folder + "_" + str(args.model_no) + "_" + str(batch_idx) + ".jpg")
    plt.show()
    #batch_idx = batch_idx + 1
    #if batch_idx == 1: break
