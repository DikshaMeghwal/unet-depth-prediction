import matplotlib
import argparse
from PIL import Image

import torch
import matplotlib.pyplot as plt

from tiny_unet import UNet
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch depth prediction evaluation script')
parser.add_argument('model_folder', type=str, metavar='F',
                    help='In which folder have you saved the model')
parser.add_argument('--path', type=str, default='data', metavar='D',
                    help="image file path")
parser.add_argument('--model_no', type=int, default = 200, metavar='N',
                    help='Which model no to evaluate (default: 1(first model))')

args = parser.parse_args()

from data import output_height, output_width

state_dict = torch.load("models/" + args.model_folder + "/model_" + str(args.model_no) + ".pth")

model = UNet()

model.load_state_dict(state_dict)
model.eval()

img = Image.open(args.path)
img = img.resize((64,64))
img_np = np.asarray(img)
img_t = torch.from_numpy(img_np)
img_t = img_t.view(1, 3, 64, 64)
img_t = img_t.float()
output = model(img_t)
output = output.detach().numpy()
print(output.shape)
plt.imsave('output.png', np.transpose(output[0][0], (0, 1)))
