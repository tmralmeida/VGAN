from torchmetrics import FID
import torch
from utils.vgan import generate_samples

def denormalize(img, mean, std):
    return (img * std + mean)*255

def evaluate(fid, fake_imgs, real_imgs, ls = 256):
    fake_imgs = denormalize(fake_imgs, 0.5, 0.5).type(torch.uint8)
    real_imgs = denormalize(real_imgs, 0.5, 0.5).type(torch.uint8)
    
    fid.update(fake_imgs, real=False)
    fid.update(real_imgs, real=True)
    
    res = fid.compute()
    return res
    