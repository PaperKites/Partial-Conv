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



def evaluate(model, dataset, device, filename, batch):

    # initialize multiple empty lists (tuples)
    mask,image,gt = ([] for _ in range(3))

    if batch == 0 :
        # run the code on all the slices in the volume
        for Dim in range(3):
            Iteration=Trimmed.shape[Dim]
            for i in range(Iteration):
                if Dim=0:
                    gt = (*gt,img_transform(nump2img(Trimmed[i,:,:])))
                    mask = (*mask,mask_transform(nump2mask(Holes[i,:,:])))
                    image= (*image,mask[i]*gt[i])
                elif Dim=1:
                    gt = (*gt,img_transform(nump2img(Trimmed[:,i,:])))
                    mask = (*mask,mask_transform(nump2mask(Holes[:,i,:])))
                    image= (*image,mask[i]*gt[i])
                elif Dim=2:
                    gt = (*gt,img_transform(nump2img(Trimmed[:,:,i])))
                    mask = (*mask,mask_transform(nump2mask(Holes[:,:,i])))
                    image= (*image,mask[i]*gt[i])

            image = torch.stack(image)
            mask = torch.stack(mask)
            gt = torch.stack(gt)
            with torch.no_grad():
                output, _ = model(image.to(device), mask.to(device))
            output = output.to(torch.device('cpu'))
            Result = torch.stack(output,dim=4)

    elif batch == 1 :

        Result=()
        # run the code on all the slices in the volume
        for Dim in range(3):
            Iteration=Trimmed.shape[Dim]
            for BatchNumber in range(math.ceil(Iteration/16)):
                for temp in range(16)
                    index=(BatchNumber*16)+temp
                    if Dim=0:
                        gt = (*gt,img_transform(nump2img(Trimmed[index,:,:])))
                        mask = (*mask,mask_transform(nump2mask(Holes[index,:,:])))
                        image= (*image,mask[temp]*gt[temp])
                    elif Dim=1:
                        gt = (*gt,img_transform(nump2img(Trimmed[:,index,:])))
                        mask = (*mask,mask_transform(nump2mask(Holes[:,index,:])))
                        image= (*image,mask[temp]*gt[temp])
                    elif Dim=2:
                        gt = (*gt,img_transform(nump2img(Trimmed[:,:,index])))
                        mask = (*mask,mask_transform(nump2mask(Holes[:,:,index])))
                        image= (*image,mask[temp]*gt[temp])

                image = torch.stack(image)
                mask = torch.stack(mask)
                gt = torch.stack(gt)
                with torch.no_grad():
                    output, _ = model(image.to(device), mask.to(device))
                output = output.to(torch.device('cpu'))
                Result = Result+torch.unbind(output,0)
            ThreeVolumes=torch.stack(Result,dim=4)

        Result=ThreeVolumes


    ###########change the RBG conversion func to ensure that all dims have the same elements
    Sum=torch.add(Result[:,1,:,:,0],Result[:,1,:,:,1],Result[:,1:,:,:,2])
    Avg=torch.div(Sum,3)

    # fix the dimensions (DxWxH to WxDxD) note: use the following link for 3D arch https://stackoverflow.com/questions/44841654/no-n-dimensional-tranpose-in-pytorch
    Volume= Avg.transpose(0,2)

    scipy.io.savemat('Volume',Volume)
