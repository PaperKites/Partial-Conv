import torch
import opt

# 0   1   2   3   4
#(B x C x D x H x W)
#(B x W x H x D x C)

def unnormalize(x):
    x = x.permute(0,4,3,2,1)
    x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    x = x.permute(0,4,3,2,1)
    return x
