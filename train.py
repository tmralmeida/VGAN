import argparse
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from models.vgan import *
from utils.vgan import *
from utils.common import evaluate, FER2013GAN
from torchmetrics import FID
import os
from math import sqrt


parser = argparse.ArgumentParser(description="Train VGAN")

parser.add_argument(
    "--dataset",
    "-ds",
    type=str,
    default="CIFAR-10",
    required=False,
    help="Dataset name",
    choices=["CIFAR-10", "FER-13"]
)

parser.add_argument(
    "--model",
    type=str,
    default="VGAN",
    required=False,
    choices=["VGAN", "VGAN-GP"]
)

parser.add_argument(
    "--batch_size",
    "-bs",
    type=int,
    default=128,
    required=False,
    help="Batch size"
)

parser.add_argument(
    "--num_workers",
    "-nw",
    type=int,
    default=8,
    required=False,
    help="Number of workers"
)

parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    required=True,
    help="Epochs"
)

parser.add_argument(
    "--lr_gen",
    "-lg",
    type=int,
    default=1e-4,
    required=False,
    help="Learning rate for the generator"
)


parser.add_argument(
    "--lr_disc",
    "-ld",
    type=float,
    default=1e-4,
    required=False,
    help="Learning rate for the discriminator"
)

parser.add_argument(
    "--ic",
    type=float,
    default=0.1,
    required=False,
    help="Constraint on the MI"
)

parser.add_argument(
    "--beta",
    type=float,
    default=0.,
    required=False,
    help="Initial value for beta"
)

parser.add_argument(
    "--alpha",
    type=float,
    default=10e-5,
    required=False,
    help="Dual step size"
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="logs",
    required=False,
    help="Path to save results"
)

parser.add_argument(
    "--nimgs_save",
    type=int,
    default="16",
    required=False,
    help="Number of images to save in grid mode"
)


args = parser.parse_args()

comm_tf = [transforms.ToTensor()]
# loading dataset
if args.dataset == "CIFAR-10":
    comm_tf += [transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    train_ds = datasets.CIFAR10(root = "./data", 
                                train = True,
                                download = False if os.path.isdir("data/cifar-10-batches-py") else True,
                                transform = transforms.Compose(comm_tf))
elif args.dataset == "FER-13":
    comm_tf += [transforms.Resize((32,32))]
    train_ds = FER2013GAN(root = "./data/fer13/train/",
                          transform = transforms.Compose(comm_tf))
else:
    raise ValueError(f"{args.dataset} not implemented yet")

inp_size = (3, 32, 32)

# dataloader
train_dl = torch.utils.data.DataLoader(train_ds,
                                       args.batch_size,
                                       shuffle = True, 
                                       num_workers = args.num_workers,
                                       pin_memory = True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
if args.model == "VGAN":
    tensor_size = (args.batch_size, *inp_size)
    gen = Generator(tensor_size).to(device)
    disc = Discriminator(inp_size[0], device).to(device)

# optimizers
opt_d = torch.optim.RMSprop(disc.parameters(), lr = args.lr_disc)
opt_g = torch.optim.RMSprop(gen.parameters(), lr = args.lr_gen)
metric = FID().to(device)
writer = SummaryWriter()

for epoch in range(args.epochs):
    losses_g, losses_d, beta = [],[], args.beta
    with tqdm(train_dl, unit="batch") as tepoch:
        for num_iter,batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}")
            real_imgs, _ = batch
            real_imgs, bs = real_imgs.to(device), real_imgs.shape[0]
            loss_d, loss_kl = train_discriminator(disc, gen, real_imgs, opt_d, beta, device, bs = bs) 
            loss_g = train_generator(disc, gen, real_imgs, opt_g, device, bs = bs)
            beta = max(0., beta + args.alpha * loss_kl)
            losses_d.append(loss_d)
            losses_g.append(loss_g)
            mean_ld = sum(losses_d)/(len(losses_d))
            mean_lg = sum(losses_g)/(len(losses_g))
            tepoch.set_postfix(l_d = mean_ld,l_g = mean_lg)
            writer.add_scalar('l_g', mean_lg, num_iter)
            writer.add_scalar('l_d', mean_ld, num_iter)
            writer.add_scalar('beta', beta, num_iter)
        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            samples_tboard = generate_samples(gen, device, epoch, os.path.join(args.save_dir, args.dataset), n_imgs = args.nimgs_save)
        fake_imgs = generate_samples(gen, device, n_imgs = bs).to(device)
        writer.add_image("Fake image", make_grid(samples_tboard, nrow=int(sqrt(args.nimgs_save)), padding = 1), epoch)
        fid = evaluate(metric, fake_imgs, real_imgs, device)
        print("FID=", fid.item())
        writer.add_scalar("fid", fid.item(), epoch)
        
writer.close()
        
torch.save(gen.state_dict(), os.path.join(args.save_dir, f'{args.dataset}/gen.pth'))
torch.save(disc.state_dict(), os.path.join(args.save_dir, f'{args.dataset}/disc.pth'))