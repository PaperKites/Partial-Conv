import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
import scipy.io
from util.image import unnormalize
import random


def evaluate(model, dataset, device, filename):
    # image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image, mask, gt = zip(*[dataset[random.randrange(0, len(dataset))]]) #pick a test data randomly
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))

    image = unnormalize(image)
    output = unnormalize(output)
    gt = unnormalize(gt)

    grid = make_grid(torch.cat((image[0,:,:,4,:], mask[0,:,:,4,:], output[0,:,:,4,:], gt[0,:,:,4,:]), dim=0)) #(B x W x H x D x C)
    save_image(grid, filename+'.jpg')
    scipy.io.savemat(filename+'.mat', output)
