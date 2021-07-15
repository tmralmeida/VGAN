import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from math import sqrt
from utils.vgan_gp import compute_grad2

def vdb_loss(preds, ys, mean, logvar):
    d_loss = torch.mean(F.binary_cross_entropy(preds, ys))
    kl_loss = torch.distributions.kl_divergence(torch.distributions.Normal(mean, torch.sqrt(logvar.exp())), torch.distributions.Normal(0,1)).mean()
    return d_loss, kl_loss


def train_discriminator(discriminator, generator, real_imgs, optimizer, beta, device, bs, model = "VGAN", ls = 256, w_gp = 10.):
    optimizer.zero_grad()
    real_ys = torch.ones((bs, 1), device = device)
    fake_ys = torch.zeros((bs, 1), device = device)
    
    real_imgs.requires_grad_()
    reg = 0.
    
    # Evaluate on real images
    real_preds, mean, logvar = discriminator(real_imgs)
    real_loss, real_kl = vdb_loss(real_preds, real_ys, mean, logvar)
    real_loss.backward(retain_graph=True)
    if model == "VGAN-GP":
        reg += w_gp * compute_grad2(real_preds, real_imgs).mean()
    
    # Evaluate on fake images
    z = torch.randn((bs, ls), device = device)
    fake_imgs = generator(z)
    fake_imgs.requires_grad_()
    fake_preds, mean, logvar = discriminator(fake_imgs)
    fake_loss, fake_kl = vdb_loss(fake_preds, fake_ys, mean, logvar)
    fake_loss.backward(retain_graph=True)
    
    total_loss = (real_loss + fake_loss)
    avg_kl = 0.5 * (real_kl + fake_kl)
    reg += beta * avg_kl
    reg.backward()
    optimizer.step()
    
    return total_loss.item(), avg_kl.item()


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

def generate_samples(gen, device, *save_info, ls = 256, n_imgs = 64, log_dir = "logs"):
    if len(save_info) == 2:
        epoch = save_info[0]
        log_dir = save_info[1]
        os.makedirs(log_dir, exist_ok=True)
    gen.eval()
    z = torch.randn((n_imgs,ls), device = device)
    with torch.no_grad():
        fake_img = gen(z).to("cpu")
    if len(save_info) == 2:
        save_image(fake_img, os.path.join(log_dir, f"ep_{epoch}.png"), nrow=int(sqrt(n_imgs)))
    return fake_img