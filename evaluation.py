import torch
from torchvision.utils import make_grid, save_image
import pytorch_ssim
from loss import InpaintingLoss
from net import VGG16FeatureExtractor
import opt
from util.image import unnormalize
from tensorboardX import SummaryWriter



def evaluate(model, dataset, device, filename, epoch='', phase='test'):
    SSIM=0
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)
    SSIM_Accuracy = pytorch_ssim.SSIM()

    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))

    loss_dict = criterion(image, mask, output, gt)
    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value

    SSIM += SSIM_Accuracy(gt, output.detach())

    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

    if phase == 'eval':
        log = open("eval_log.txt","w+")
        with open(log, "a") as log_file:
            log_file.write('Epoch: [{0}]\t'
            'Loss {3:.4f}\n'.format(
            (epoch + 1), loss,))

        writer = SummaryWriter(log_dir='./')
        writer.add_scalar('Loss', loss, (epoch+1))
        writer.add_scalar('SSIM Accuracy', SSIM.item(), (epoch+1))
        writer.flush()
