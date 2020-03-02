import torch
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
import os

from util.ssim import SSIM
from model.loss import InpaintingLoss
from model.net import VGG16FeatureExtractor
import model.opt
from util.image import unnormalize

def evaluate(model, dataset, device, filename, epoch ,log_dir):


    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)
    SSIM_Accuracy = SSIM()
    ssim = 0

    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))

    loss_dict = criterion(image.to(device), mask.to(device), output, gt.to(device))
    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value

    ssim += SSIM_Accuracy(gt.to(device), output)

    output = output.to(torch.device('cpu'))

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                    unnormalize(gt)), dim=0))
    save_image(grid, filename)


    log_dir = log_dir + '/eval_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log = os.path.join(log_dir, 'eval_log.txt')
    with open(log, "a") as log_file:
        log_file.write('Epoch: [{0}]\t'
        'Accuracy: {1:.4f}\t'
        'Loss {2:.4f}\n'.format(
        epoch, ssim, loss))

    writer = SummaryWriter(log_dir = log_dir)
    writer.add_scalar('Loss/Loss', loss, (epoch+1))
    writer.add_scalar('Accuracy/SSIM accuracy', ssim.item(), (epoch+1))
    writer.close()
