import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
import opt
from PIL import Image
import numpy as np
import cv2
from util.image import unnormalize
import scipy.io

Holes = scipy.io.loadmat('data.mat')['Holes_Vol']
Trimmed = scipy.io.loadmat('data.mat')['Trimmed_Vol']

Iteration=8

size = (256, 256)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])


def nump2img(nump):
    nump= Image.fromarray(nump)
    return nump.convert('RGB')


def nump2mask(nump):
    nump=1-nump
    nump = Image.fromarray(nump*255)
    return nump.convert('RGB')



def evaluate(model, dataset, device, filename):

    # initialize multiple empty lists (tuples)
    mask,image,gt = ([] for _ in range(3))

    # Change the dimension if needed
    for i in range(Iteration):
        gt = (*gt,img_transform(nump2img(Trimmed[:,i,:])))
        mask = (*mask,mask_transform(nump2mask(Holes[:,i,:])))
        image= (*image,mask[i]*gt[i])

    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))

    # grid = make_grid(torch.cat((unnormalize(image), mask, unnormalize(output), unnormalize(gt)), dim=0),Iteration)
    
    grid = make_grid(torch.cat((unnormalize(output)[0:1] ,unnormalize(gt)[0:1] ), dim=0),1)
    save_image(grid, filename)
