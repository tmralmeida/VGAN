import torch
from torch.utils.data import Dataset
from torchmetrics import FID
from PIL import Image
import os
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
    
class FER2013GAN(Dataset):
    """FER dataset object

    Args:
        root (str): path
        transforms (torchvision.transforms): preprocessing transformations
    """
    def __init__(self,
                 root,
                 transform = None):
        self.root = root
        self.transform = transform
        self.map_idx = {"sad":0,
                        "surprised":1,
                        "neutral":2,
                        "disgusted":3,
                        "happy":4,
                        "angry":5,
                        "fearful":6}
        self.__preprocess()
        
    def __preprocess(self):
        self.paths = []
        for dirpath,_,filenames in os.walk(self.root):
            for f in filenames:
                 self.paths.append((os.path.abspath((os.path.join(dirpath, f))), self.map_idx[dirpath.split("train/")[-1]])) 
        
    def __getitem__(self, idx):
        img, lbl = Image.open(self.paths[idx][0]).convert("RGB"), self.paths[idx][1]
        if self.transform:
            img = self.transform(img)
        return img, lbl
        
        
    def __len__(self):
        return len(self.paths)