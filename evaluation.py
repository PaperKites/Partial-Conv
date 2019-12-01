import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
import opt
from PIL import Image, ImageOps
import numpy as np
import cv2
from util.image import unnormalize
import scipy.io
import math

Holes = scipy.io.loadmat('data.mat')['Holes_Vol']
Trimmed = scipy.io.loadmat('data.mat')['Trimmed_Vol']

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



def evaluate(model, dataset, device, filename, batch=1):

    # initialize multiple empty lists (tuples)
    mask,image,gt = ([] for _ in range(3))

    if batch == 0 :
        # run the code on all the slices in the volume
        for Dim in range(3):
            Iteration=Trimmed.shape[Dim]
            for i in range(Iteration):
                if Dim==0:
                    gt = (*gt,img_transform(nump2img(Trimmed[i,:,:])))
                    mask = (*mask,mask_transform(nump2mask(Holes[i,:,:])))
                    image= (*image,mask[i]*gt[i])
                elif Dim==1:
                    gt = (*gt,img_transform(nump2img(Trimmed[:,i,:])))
                    mask = (*mask,mask_transform(nump2mask(Holes[:,i,:])))
                    image= (*image,mask[i]*gt[i])
                elif Dim==2:
                    gt = (*gt,img_transform(nump2img(Trimmed[:,:,i])))
                    mask = (*mask,mask_transform(nump2mask(Holes[:,:,i])))
                    image= (*image,mask[i]*gt[i])

            image = torch.stack(image)
            mask = torch.stack(mask)
            gt = torch.stack(gt)
            with torch.no_grad():
                output, _ = model(image.to(device), mask.to(device))
            output = output.to(torch.device('cpu'))
            result = torch.stack(output,dim=4)

    elif batch == 1 :

        min=Trimmed.shape[0]
        for  Dim in range(3):
            if Trimmed.shape[Dim]<min:
                min=Trimmed.shape[Dim]

        ThreeVolumes = []
        for Dim in range(3):
            Result=torch.empty((0,3,256,256))
            Iteration=Trimmed.shape[Dim]

            for BatchNumber in range(math.ceil(min/16)):
                mask,image,gt = ([] for _ in range(3))
                for temp in range(16):
                    index=(BatchNumber*16)+temp
                    if index < Iteration :
                        if Dim==0:
                            gt = (*gt,img_transform(nump2img(Trimmed[index,:,:])))
                            mask = (*mask,mask_transform(nump2mask(Holes[index,:,:])))
                            image= (*image,mask[temp]*gt[temp])
                        elif Dim==1:
                            gt = (*gt,img_transform(nump2img(Trimmed[:,index,:])))
                            mask = (*mask,mask_transform(nump2mask(Holes[:,index,:])))
                            image= (*image,mask[temp]*gt[temp])
                        elif Dim==2:
                            gt = (*gt,img_transform(nump2img(Trimmed[:,:,index])))
                            mask = (*mask,mask_transform(nump2mask(Holes[:,:,index])))
                            image= (*image,mask[temp]*gt[temp])

                image = torch.stack(image)
                mask = torch.stack(mask)
                gt = torch.stack(gt)
                with torch.no_grad():
                    output, _ = model(image.to(device), mask.to(device))
                output = output.to(torch.device('cpu'))
                Result = torch.cat((Result,output), 0)
            ThreeVolumes.append(Result[:min])
        result=torch.stack(ThreeVolumes,dim=2)
        print(result.size())

    result = unnormalize(result)
    print(result.size())
    Sum=torch.add(result[:,:,:,0,1],torch.add(result[:,:,:,1,1],result[:,::,:,2,1]))
    print(Sum.size())
    Avg=torch.div(Sum,3)
    print(Avg.size())
    scipy.io.savemat('Volume.mat',{'Volume': Avg})
