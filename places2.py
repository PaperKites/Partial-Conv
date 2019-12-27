import random
import torch
from PIL import Image
from glob import glob
import PIL.ImageOps
import os

class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.split=split
        # use about 8M images in the challenge dataset
        if split == 'train':
            self.paths = glob('{:s}/data_large/*'.format(img_root))
        else:
            self.paths = glob('{:s}/{:s}_large/*'.format(img_root, split))

        self.mask_paths = glob('{:s}/*'.format(mask_root))
        self.N_mask = len(self.mask_paths)
        self.N_img = len(self.paths)

    def __getitem__(self, index):
        if self.split == 'train':
            gt_img = Image.open(self.paths[index])
        else:
            gt_img = Image.open(self.paths[random.randint(0, self.N_img - 1)])

        if gt_img.size[1] != 256 or gt_img.size[0] != 256:
            gt_img=gt_img.resize((256,256))

        gt_img = self.img_transform(gt_img.convert('RGB'))
        index=random.randint(0, self.N_mask - 1)
        mask = Image.open(self.mask_paths[index])
        if "mask" in os.path.basename(self.mask_paths[index]):
            mask =  PIL.ImageOps.invert(mask.convert('RGB'))
        else:
            mask=mask.convert("RGB")
        if mask.size[1] != 256 or mask.size[0] != 256:
            mask=mask.resize((256,256))

        mask = self.mask_transform(mask)
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
