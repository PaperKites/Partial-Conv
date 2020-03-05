import torch
from util.image import unnormalize
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import scipy.io
import math
import argparse

from model.net import PConvUNet
from util.io import load_ckpt
from model import opt

parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type = str, default = None, help = 'Pre-trained model filename')
parser.add_argument('--image_size', type = int, default = 256, help = 'Image dimensions')
parser.add_argument('--volume_name', type = str, default = 'data.mat', help = 'Volume filename')
parser.add_argument('--batch_size', type = int, default = 16, help = 'Mini-batch size')

args = parser.parse_args()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Used device:', device)

# Change it as needed (filename)[Variable_Name]
Holes = scipy.io.loadmat(args.volume_name)['Holes_Vol']
Volume = scipy.io.loadmat(args.volume_name)['Trimmed_Vol']

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()

# convert numpy array to image
def nump2img(nump):
    nump = Image.fromarray(nump)
    return nump

# convert numpy array to mask
def nump2mask(nump):
    nump = 1 - nump #Invert binary array
    nump = Image.fromarray(nump * 255)
    return nump



size = (args.image_size, args.image_size) # The model is trained on 256x256 images (Places2 Dataset)
img_transform = transforms.Compose(
    [transforms.Resize(size=size,interpolation=Image.NEAREST), transforms.Grayscale(3),
    transforms.ToTensor(),   #transforms.ToTensor() convert 0-255 to 0-1 implicitly
    transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size,interpolation=Image.NEAREST), transforms.Grayscale(3),
    transforms.ToTensor()])

X,Y,Z = Volume.shape

# Iterate over the 3 axis
for Dim in range(3):
    Result = torch.zeros((0,3,)+size)
    Iteration = Volume.shape[Dim]

    for BatchNumber in range(math.ceil(Iteration/args.batch_size)):  # Batch=16, so iterate NumberOfSlides/16 times
        mask,image,gt = ([] for _ in range(3))

        for temp in range(args.batch_size):
            index = (BatchNumber*args.batch_size)+temp
            if index < Iteration :
                if Dim == 0:
                    gt = (*gt,img_transform(nump2img(Volume[index,:,:])))
                    mask = (*mask,mask_transform(nump2mask(Holes[index,:,:])))
                    image = (*image,mask[temp]*gt[temp])
                elif Dim == 1:
                    gt = (*gt,img_transform(nump2img(Volume[:,index,:])))
                    mask = (*mask,mask_transform(nump2mask(Holes[:,index,:])))
                    image = (*image,mask[temp]*gt[temp])
                elif Dim == 2:
                    gt = (*gt,img_transform(nump2img(Volume[:,:,index])))
                    mask = (*mask,mask_transform(nump2mask(Holes[:,:,index])))
                    image = (*image,mask[temp]*gt[temp])

        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)
        with torch.no_grad():
            output, _ = model(image.to(device), mask.to(device))
        output = output.to(torch.device('cpu')) #torch.Size([16, 3, 256, 256]) [B,C,W,H]
        Result = torch.cat((Result,output), 0)  # initially i=0; torch.Size([i+=16, 3, 256, 256]) [NumOfBatches,C,W,H]

    if Dim == 0:
        Result = unnormalize(Result)
        First_Dim = F.interpolate(Result, size=(Y,Z), mode='bicubic')  #The resize operation on tensor.
    elif Dim == 1:
        Result = unnormalize(Result)
        Sec_Dim = F.interpolate(Result, size=(X,Z), mode='bicubic')  #The resize operation on tensor.
    elif Dim == 2:
        Result = unnormalize(Result)
        Third_Dim = F.interpolate(Result, size=(X,Y), mode='bicubic')  #The resize operation on tensor.

First_Dim = First_Dim.mul(255).clamp_(0, 255).permute(0,2,3,1)    #(B x W x H x C)
Sec_Dim = Sec_Dim.mul(255).clamp_(0, 255).permute(2,0,3,1)
Third_Dim = Third_Dim.mul(255).clamp_(0, 255).permute(2,3,0,1)

Sum = torch.add(First_Dim,torch.add(Sec_Dim ,Third_Dim))
Avg = torch.div(Sum,3)

scipy.io.savemat('output_' + args.volume_name, {"first": First_Dim[...,0].numpy(),
"sec": Sec_Dim[...,0].numpy(), "third": Third_Dim[...,0].numpy(), "Avg": Avg[...,0].numpy()})
