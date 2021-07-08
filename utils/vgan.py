import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from math import sqrt

def vdb_loss(preds, ys, mean, logvar, beta):
    d_loss = torch.mean(F.binary_cross_entropy(preds, ys))
    kl_loss = torch.distributions.kl_divergence(torch.distributions.Normal(mean, torch.sqrt(logvar.exp())), torch.distributions.Normal(0,1)).mean()
    total_loss = d_loss + beta * kl_loss 
    return total_loss, kl_loss.detach()


def train_discriminator(discriminator, generator, real_imgs, optimizer, beta, device, bs, ls = 256):
    optimizer.zero_grad()
    real_ys = torch.ones((bs, 1), device = device)
    fake_ys = torch.zeros((bs, 1), device = device)
    
    # Evaluate on real images
    real_preds, mean, logvar = discriminator(real_imgs)
    real_loss, real_kl = vdb_loss(real_preds, real_ys, mean, logvar, beta)
    
    # Evaluate on fake images
    z = torch.randn((bs, ls), device = device)
    fake_imgs = generator(z)
    fake_preds, mean, logvar = discriminator(fake_imgs)
    fake_loss, fake_kl = vdb_loss(fake_preds, fake_ys, mean, logvar, beta)
    
    total_loss = real_loss + fake_loss
    total_loss.backward()
    optimizer.step()
    total_kl = real_kl + fake_kl
    
    return total_loss.item(), total_kl.item()


def train_generator(discriminator, generator, real_imgs, optimizer, device, bs = 128, ls = 256):
    optimizer.zero_grad()
    
    ys = torch.ones((bs, 1), device = device)
    
    # Fooling the discriminator
    z = torch.randn((bs, ls), device = device)
    fake_imgs = generator(z)
    preds, _, _ = discriminator(fake_imgs)
    loss = F.binary_cross_entropy(preds, ys)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def generate_samples(epoch, gen, device, ls = 256, nimgs_save = 64, log_dir = "logs"):
    gen.eval()
    os.makedirs(log_dir, exist_ok=True)
    
    z = torch.randn((nimgs_save,ls), device = device)
    with torch.no_grad():
        fake_img = gen(z).to("cpu")
    save_image(fake_img, os.path.join(log_dir, f"ep_{epoch}.png"), nrow=int(sqrt(nimgs_save)))