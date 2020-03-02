import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import time
from datetime import datetime
import time
from datetime import datetime
import sys

from model import opt
from evaluation import evaluate
from model.loss import InpaintingLoss
from model.net import PConvUNet, VGG16FeatureExtractor
from util.Dataset import Dataset
from util.io import load_ckpt, save_ckpt
from util.ssim import SSIM


# training options
parser = argparse.ArgumentParser()
parser.add_argument('--root', type = str, default = './data', help = 'Dataset directory')
parser.add_argument('--save_dir', type = str, default = './snapshots', help = 'Model save directory')
parser.add_argument('--log_dir', type = str, default = '/logs', help = 'logging directory')
parser.add_argument('--lr', type = float, default = 2e-4, help = 'Training learning rate')
parser.add_argument('--lr_finetune', type = float, default = 5e-5, help = 'Fine-tuning learning rate')
parser.add_argument('--epochs', type = int, default = 50, help = 'Number of total epochs to run')
parser.add_argument('--batch_size', type = int, default = 8, help = 'Mini-batch size')
parser.add_argument('--n_threads', type = int, default = 4, help = 'Number of data loading workers')
parser.add_argument('--save_model_interval', type = int, default=1, help = 'Model saving frequency')
parser.add_argument('--vis_interval', type = int, default = 1, help = 'Model evaluating frequency')
parser.add_argument('--log_interval', type = int, default = 1, help = 'Logging frequency')
parser.add_argument('--image_size', type = int, default = 256,  help = 'Image dimensions')
parser.add_argument('--resume', type = str, default = None, help = 'Pre-trained model filename')
args = parser.parse_args()


torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else sys.exit())

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

args.log_dir = '{:s}{:s}'.format(args.save_dir,args.log_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

SSIM_Accuracy = SSIM()


def main():


    writer = SummaryWriter(log_dir=args.log_dir)
    log = os.path.join(args.log_dir, 'loss.txt')
    with open(log, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    size = (args.image_size, args.image_size)
    img_tf = transforms.Compose(
            [transforms.Resize(size=size,interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
    mask_tf = transforms.Compose(
            [transforms.Resize(size=size,interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()])

    dataset_train = Dataset(args.root, img_tf, mask_tf, 'train')
    dataset_val = Dataset(args.root, img_tf, mask_tf, 'val')

    train_loader = data.DataLoader(
        dataset_train, batch_size = args.batch_size,
        shuffle = True,pin_memory = True,
        num_workers = args.n_threads)

    print('Dataset size = ' + str(len(dataset_train)))
    model = PConvUNet().to(device)

    if args.resume is not None:
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
        print('Starting from ', start_iter)


    with open(log, "a") as log_file:
        log_file.write('starting training\n')

    scheduler = StepLR(optimizer, step_size=30)

    for epoch in tqdm(range(start_iter, (args.epochs+start_iter))):

        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decay Learning Rate (adjusting learning rate)
        scheduler.step()
        model.train()

        SSIM = 0
        L1 = 0
        losses = np.zeros(5)

        end = time.time()
        for i, (image, mask, gt) in enumerate(train_loader):

             # measure data loading time
            data_time.update(time.time() - end)

            image = image.cuda(device, non_blocking=True)
            mask = mask.cuda(device, non_blocking=True)
            gt = gt.cuda(device, non_blocking=True)

            output, _ = model(image, mask)
            loss_dict = criterion(image, mask, output, gt)
            loss = 0.0
            index = 0
            for key, coef in opt.LAMBDA_DICT.items():
                value = coef * loss_dict[key]
                loss += value
                losses[index] += value
                index += 1


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
                    'Accuracy: {4:.4f}\t'
                    'Loss {3:.4f}\t'
                    'LR {5}\n'.format(
                    (epoch + 1), i, len(train_loader), loss, SSIM_Accuracy(gt.detach(), output.detach()),
                    scheduler.get_lr(), batch_time = batch_time,
                    data_time = data_time))

        if (epoch + 1) % args.log_interval == 0:
            index = 0
            for key, coef in opt.LAMBDA_DICT.items():
                writer.add_scalar('etc/loss_{:s}'.format(key), losses[index]/i, (epoch-start_iter+1))
                index += 1
            writer.flush()

        if (epoch + 1) % args.log_interval == 0:
            writer.add_scalar('Accuracy/L1 accuracy', L1.item()/i, (epoch-start_iter+1))
            writer.add_scalar('Accuracy/SSIM accuracy', SSIM.item()/i, (epoch-start_iter+1))
            writer.add_scalar('Loss/Loss', losses.sum()/i, (epoch-start_iter+1))
            writer.flush()

        if (epoch + 1) % args.save_model_interval == 0 or (epoch + 1) == (args.epochs+start_iter+1):
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, (epoch + 1)),
                      [('model', model)], [('optimizer', optimizer)], (epoch + 1))

        if (epoch + 1) % args.vis_interval == 0:
            model.eval()
            evaluate(model, dataset_val, device,
                     '{:s}/images/test_{:d}.jpg'.format(args.save_dir, (epoch + 1)),(epoch-start_iter+1),args.log_dir)

        print( "Epoch:", (epoch-start_iter+1), "Loss: {:.4f}".format(losses.sum()/i),"L1: {:.4f}".format(L1.item()/i) , "SSIM: {:.4f}".format(SSIM.item()/i), 'Time:',datetime.now().strftime("%H:%M:%S"))

    writer.close()

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
