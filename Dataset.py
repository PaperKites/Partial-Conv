import random
import torch
from PIL import Image
from glob import glob
import scipy.io


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Dataset, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('{:s}/*.mat'.format(img_root)
        else:
            self.paths = glob('{:s}/val/*mat'.format(img_root, split))

    def __getitem__(self, index):
        gt_img = scipy.io.loadmat(self.paths[index])['Trimmed_Vol']
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = scipy.io.loadmat(self.paths[index])['Holes_Vol']
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
