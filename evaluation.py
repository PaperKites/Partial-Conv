import torch
from torchvision.utils import make_grid, save_image
from util.image import unnormalize
from torchvision import transforms
import torch.nn.functional as F
import opt
from PIL import Image, ImageOps
import numpy as np
import cv2
import scipy.io
import math

Holes = scipy.io.loadmat('data.mat')['Holes_Vol']
Volume = scipy.io.loadmat('data.mat')['Trimmed_Vol']

size = (256, 256) # The model is trained on 256x256 images (Places2 Dataset)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])   #transforms.ToTensor() convert 0-255 to 0-1 implicitly

# convert numpy array to image
def nump2img(nump):
    nump= Image.fromarray(nump)
    return nump.convert('RGB')

# convert numpy array to image, but for masks
def nump2mask(nump):
    nump=1-nump #Invert binary array
    nump = Image.fromarray(nump*255)
    return nump.convert('RGB')

def evaluate(model, dataset, device, filename):

    ThreeVolumes = []
    X,Y,Z =Volume.shape

    #highest dimension
    max=Volume.shape[0]
    for  Dim in range(3):
        if Volume.shape[Dim]>max:
            max=Volume.shape[Dim]

    # Iterate over the 3 axis
    for Dim in range(3):
        Result=torch.zeros((0,3,)+size)
        Iteration=Volume.shape[Dim]

        for BatchNumber in range(math.ceil(Iteration/16)):  # Batch=16, so iterate NumberOfSlides/16 times
            mask,image,gt = ([] for _ in range(3))

            for temp in range(16):
                index=(BatchNumber*16)+temp
                if index < Iteration :
                    if Dim==0:
                        gt = (*gt,img_transform(nump2img(Volume[index,:,:])))
                        mask = (*mask,mask_transform(nump2mask(Holes[index,:,:])))
                        image= (*image,mask[temp]*gt[temp])
                    elif Dim==1:
                        gt = (*gt,img_transform(nump2img(Volume[:,index,:])))
                        mask = (*mask,mask_transform(nump2mask(Holes[:,index,:])))
                        image= (*image,mask[temp]*gt[temp])
                    elif Dim==2:
                        gt = (*gt,img_transform(nump2img(Volume[:,:,index])))
                        mask = (*mask,mask_transform(nump2mask(Holes[:,:,index])))
                        image= (*image,mask[temp]*gt[temp])

            image = torch.stack(image)
            mask = torch.stack(mask)
            gt = torch.stack(gt)
            with torch.no_grad():
                output, _ = model(image.to(device), mask.to(device))
            output = output.to(torch.device('cpu')) #torch.Size([16, 3, 256, 256]) [B,C,W,H]
            Result = torch.cat((Result,output), 0)  # initially i=0; torch.Size([i+=16, 3, 256, 256]) [NumOfBatches,C,W,H]

        if Dim==0:
            Result = unnormalize(Result)
            out0 = F.interpolate(Result, size=(Y,Z))  #The resize operation on tensor.
            ThreeVolumes.append(out0)
        elif Dim==1:
            Result = unnormalize(Result)
            out1 = F.interpolate(Result, size=(X,Z))  #The resize operation on tensor.
            ThreeVolumes.append(out1)
        elif Dim==2:
            Result = unnormalize(Result)
            out2 = F.interpolate(Result, size=(X,Y))  #The resize operation on tensor.
            ThreeVolumes.append(out2)

    out0 = out0.permute(0,2,3,1)    #(B x W x H x C)
    out1 = out1.permute(2,0,3,1)
    out2 = out2.permute(2,3,0,1)

    Sum=torch.add(out0,torch.add(out1 ,out2))  #torch.Size([626, 256, 256, 3])
    Avg=torch.div(Sum,3)

    for i in range(0,10,2):
        Temp=Avg.mul(255).clamp_(0, 255).numpy()
        img = Image.fromarray((Temp[:,i,:,:]).astype(np.uint8)).convert('L')
        img.save('{:d}Avg.jpg'.format(i + 1))

        # Temp1=out1.mul(255).clamp_(0, 255).numpy()
        # img1 = Image.fromarray((Temp1[:,i,:,:]).astype(np.uint8)).convert('L')
        # img1.save('{:d}.jpg'.format(i + 1))
