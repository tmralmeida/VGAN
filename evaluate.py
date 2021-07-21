import argparse
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from models.vgan import *
from utils.common import *
from utils.datasets import * 
from torchmetrics import FID
from torch.utils.data import Subset

parser = argparse.ArgumentParser(description="Evaluate VGAN")

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
    "--discriminator_path",
    "-dp",
    type=str,
    required=True,
    help="Relative path to the discriminator saved model"
)

parser.add_argument(
    "--generator_path",
    "-gp",
    type=str,
    required=True,
    help="Relative path to the generator saved model"
)

parser.add_argument(
    "--nsamples",
    "-ns",
    type=int,
    default=10000,
    required=False,
    help= "Number of samples to evaluate"
)


parser.add_argument(
    "--num_workers",
    "-nw",
    type=int,
    default=8,
    required=False,
    help="Number of workers"
)

args = parser.parse_args()


comm_tf = [transforms.ToTensor()]
# loading dataset
if args.dataset == "CIFAR-10":
    comm_tf += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    test_ds = datasets.CIFAR10(root = "./data", 
                               train = True,
                               download = False if os.path.isdir("data/cifar-10-batches-py") else True,
                               transform = transforms.Compose(comm_tf))
elif args.dataset == "FER-13":
    comm_tf += [transforms.Resize((32,32))]
    test_ds = FER2013GAN(root = "./data/fer13/train/",
                         transform = transforms.Compose(comm_tf))
else:
    raise ValueError(f"{args.dataset} not implemented yet")

inp_size = (3, 32, 32)

samp_dl = Subset(test_ds, torch.arange(args.nsamples))
# dataloader
test_dl = torch.utils.data.DataLoader(samp_dl,
                                      args.batch_size,
                                      shuffle = True, 
                                      num_workers = args.num_workers,
                                      pin_memory = True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
if args.model == "VGAN" or args.model == "VGAN-GP":
    tensor_size = (args.batch_size, *inp_size)
    gen = Generator(tensor_size).to(device)
    disc = Discriminator(inp_size[0], device).to(device)
    
g_state_dict = torch.load(args.generator_path)
d_state_dict = torch.load(args.discriminator_path)
gen.load_state_dict(g_state_dict)
disc.load_state_dict(d_state_dict)


metric = FID().to(device)
with tqdm(test_dl, unit="batch") as titer:
    for batch in titer:
        real_imgs, _ = batch
        bs = real_imgs.shape[0]
        fake_imgs = generate_samples(gen, device, n_imgs = bs)
        fid = evaluate(metric, fake_imgs.to(device), real_imgs.to(device))
        result = "{:.2F}".format(fid.detach().item())
        titer.set_postfix(FID = result)