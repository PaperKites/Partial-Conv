import random
import torch
from PIL import Image, ImageOps
from glob import glob
import scipy.io
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_root, img_transform, mask_transform,
                 split='train'):
        super(Dataset, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.split = split

        if split == 'train':
            self.paths = glob('{:s}/{:s}/*'.format(img_root, split))
        else:
            self.paths = glob('{:s}/{:s}/*'.format(img_root, split))

        self.mask_paths = glob('{:s}/mask/*'.format(img_root))

    def __getitem__(self, index):

        if self.split == 'train':
            gt_img = Image.open(self.paths[index])
        else:
            gt_img = Image.open(self.paths[random.randint(0, len(self.paths) - 1)])

        gt_img = self.img_transform(gt_img.convert('RGB'))

        index = random.randint(0, len(self.mask_paths) - 1)
        mask = Image.open(self.mask_paths[index])

        # The if statment below can be removed
        # The mask dataset has 2 types, one of them needs to get inverted. (because the black pixels are holes)
        # so any image with a filename containts 'mask' will get inverted
        if "mask" in os.path.basename(self.mask_paths[index]):
            mask = ImageOps.invert(mask.convert('RGB'))
        else:
            mask = mask.convert("RGB")

        mask = self.mask_transform(mask)
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
