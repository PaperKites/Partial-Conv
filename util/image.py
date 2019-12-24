import torch
import opt

# for 3D data
# def unnormalize(x):
#     x = x.permute(0,4,3,2,1)                                      # 0   1   2   3   4
#     x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)        #(B x C x D x H x W)
#     x = x.permute(0,4,3,2,1)                                      #(B x W x H x D x C)
#     return x

def unnormalize(x):
    x = x.transpose(1, 3)
    x = (x * torch.Tensor(opt.STD)) + torch.Tensor(opt.MEAN)
    x = x.transpose(1, 3)
    return x
