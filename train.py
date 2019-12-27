import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet, VGG16FeatureExtractor
from places2 import Places2
from util.io import load_ckpt, save_ckpt
import pytorch_ssim
import time
from datetime import datetime

# training options
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--mask_root', type=str, default='./data/mask')
parser.add_argument('--save_dir', type=str, default='./snapshots')
parser.add_argument('--log_dir', type=str, default='/logs')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--num_iter', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_threads', type=int, default=8)
parser.add_argument('--save_model_interval', type=int, default=1)
parser.add_argument('--vis_interval', type=int, default=1)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
args = parser.parse_args()



torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

if not os.path.exists(args.log_dir):
    os.makedirs('{:s}/{:s}'.format(args.save_dir,args.log_dir))

SSIM_Accuracy = pytorch_ssim.SSIM()


def main():


    writer = SummaryWriter(log_dir=args.log_dir)
    log = os.path.join(args.log_dir, 'loss.txt')
    with open(log, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    size = (args.image_size, args.image_size)
    img_tf = transforms.Compose(
            [transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
    mask_tf = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()])

    dataset_train = Places2(args.root, args.mask_root, img_tf, mask_tf, 'train')
    dataset_val = Places2(args.root, args.mask_root, img_tf, mask_tf, 'val')

    train_loader = data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        shuffle=True,pin_memory=True,
        num_workers=args.n_threads)

    print('Dataset size= '+str(len(dataset_train)))
    model = PConvUNet().to(device)

    if args.finetune:
        lr = args.lr_finetune
        model.freeze_enc_bn = True
    else:
        lr = args.lr

    start_iter = 0
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

    if args.resume:
        start_iter = load_ckpt(
            args.resume, [('model', model)], [('optimizer', optimizer)])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)

    with open(log, "a") as log_file:
        log_file.write('started training\n')


    for epoch in tqdm(range(start_iter, (args.num_iter+start_iter))):

        # adjust_learning_rate(optimizer, epoch)
        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()


        SSIM=0
        L1=0
        losses=np.zeros(5)

        end = time.time()
        for i, (image, mask,gt) in enumerate(train_loader):

             # measure data loading time
            data_time.update(time.time() - end)

            image = image.cuda(device, non_blocking=True)
            mask = mask.cuda(device, non_blocking=True)
            gt = gt.cuda(device, non_blocking=True)

            output, _ = model(image, mask)
            loss_dict = criterion(image, mask, output, gt)
            loss = 0.0
            index=0
            for key, coef in opt.LAMBDA_DICT.items():
                value = coef * loss_dict[key]
                loss += value
                losses[index] += value
                index+=1


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            L1 += torch.mean(torch.abs(gt.detach() - output.detach()))
            SSIM += SSIM_Accuracy(gt.detach(), output.detach())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (epoch + 1) % args.log_interval == 0:
                with open(log, "a") as log_file:
                    log_file.write('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {3:.4f}\n'.format(
                    (epoch + 1), i, len(train_loader), loss, batch_time=batch_time,
                    data_time=data_time))

        if (epoch + 1) % args.log_interval == 0:
            index=0
            for key, coef in opt.LAMBDA_DICT.items():
                writer.add_scalar('Loss/loss_{:s}'.format(key), losses[index]/i, (epoch-start_iter+1))
                index+=1
            writer.flush()

        if (epoch + 1) % args.log_interval == 0:
            writer.add_scalar('Accuracy/L1 accuracy', L1.item()/i, (epoch-start_iter+1))
            writer.add_scalar('Accuracy/SSIM accuracy', SSIM.item()/i, (epoch-start_iter+1))
            writer.add_scalar('Accuracy/Loss accuracy', losses.sum()/i, (epoch-start_iter+1))
            writer.flush()

        if (epoch + 1) % args.save_model_interval == 0 or (epoch + 1) == (args.num_iter+start_iter+1):
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, (epoch + 1)),
                      [('model', model)], [('optimizer', optimizer)], (epoch + 1))

        if (epoch + 1) % args.vis_interval == 0:
            model.eval()
            evaluate(model, dataset_val, device,
                     '{:s}/images/test_{:d}.jpg'.format(args.save_dir, (epoch + 1)),epoch,'eval')

        print("Loss: {:.4f}".format(losses.sum()/i),"L1: {:.4f}".format(L1.item()/i) , "SSIM: {:.4f}".format(SSIM.item()/i), "Epoch:", (epoch-start_iter+1), 'Time:',datetime.now().strftime("%H:%M:%S"))

    writer.close()




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
