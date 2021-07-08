import argparse
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from models.vgan import *
from utils.vgan import *
import os


parser = argparse.ArgumentParser(description="Train VGAN")

parser.add_argument(
    "--dataset",
    "-ds",
    type=str,
    default="CIFAR-10",
    required=False,
    help="Dataset name",
    choices=["CIFAR-10", "MNIST", "FER-2013"]
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

# loading dataset
if args.dataset == "CIFAR-10":
    trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR10(root = "./data", 
                                train = True,
                                download = False if os.path.isdir("data/cifar-10-batches-py") else True,
                                transform = trans)
    inp_size = (3, 32, 32)
elif args.dataset == "MNIST":
    train_ds = datasets.MNIST(root = "./data", 
                              train = True,
                              download = False if os.path.isdir("data/MNIST") else True,
                              transform = transforms.ToTensor())
    inp_size = (1, 28, 28)
else:
    raise ValueError(f"{args.dataset} not implemented yet")

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


for epoch in range(args.epochs):
    losses_g, losses_d, beta = [],[], args.beta
    with tqdm(train_dl, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            real_imgs, _ = batch
            real_imgs, bs = real_imgs.to(device), real_imgs.shape[0]
            loss_d, loss_kl = train_discriminator(disc, gen, real_imgs, opt_d, beta, device, bs = bs) 
            loss_g = train_generator(disc, gen, real_imgs, opt_g, device, bs = bs)
            beta = max(0., beta + args.alpha * loss_kl)
            losses_d.append(loss_d)
            losses_g.append(loss_g)
            tepoch.set_postfix(loss_disc = sum(losses_d)/(len(losses_d)),loss_gen = sum(losses_g)/(len(losses_g)))
        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            generate_samples(epoch, gen, device, nimgs_save = args.nimgs_save, log_dir=os.path.join(args.save_dir, args.dataset))
        
        
torch.save(gen.state_dict(), os.path.join(args.save_dir, f'{args.dataset}/gen.pth'))
torch.save(disc.state_dict(), os.path.join(args.save_dir, f'{args.dataset}/disc.pth'))