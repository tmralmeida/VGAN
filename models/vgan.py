import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, hid_channels : int, bias : bool):
        super().__init__()
        self.shortcut = in_channels != out_channels
        self.conv_0 = nn.Conv2d(in_channels, hid_channels, 3, stride = 1, padding = 1)
        self.conv_1 = nn.Conv2d(hid_channels, out_channels, 3, stride = 1, padding = 1, bias = bias)
        if self.shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride = 1, bias = False)
    
    def forward(self, x):
        xs = x if not self.shortcut else self.conv_shortcut(x)
        x = F.leaky_relu(self.conv_0(x), 0.2)
        x = F.leaky_relu(self.conv_1(x), 0.2)
        return xs + 0.1 * x
    


class Discriminator(nn.Sequential):
    def __init__(self, in_channels: int, device : torch.device):
        super().__init__()
        self.device =device
        n_c = (64, 128, 256)
        blocks = [
            ResNetBlock(in_channels = 32, out_channels = 64, hid_channels = 32, bias = True),
            nn.AvgPool2d(3, stride = 2, padding = 1)
        ]
        for i in range(2):
            blocks += [
                ResNetBlock(in_channels = n_c[i], out_channels = n_c[i+1], hid_channels = n_c[i], bias = True),
                nn.AvgPool2d(3, stride = 2, padding = 1),
            ]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, padding = 1),
            *blocks,
                    )
        self._mean = nn.Conv2d(256, 256, kernel_size = 1)
        self._logvar = nn.Conv2d(256, 256, kernel_size = 1)
        self.classifier = nn.Linear(256 * 4 * 4, 1)
        
        
    def forward(self, x):
        x = self.encoder(x)
        
        _mean = self._mean(x).view(-1, 256*4*4)
        _logvar = self._logvar(x).view(-1, 256*4*4)
        z = (0.5 * _logvar).exp() * torch.randn(_mean.size(), device = self.device) + _mean
        x = torch.sigmoid(self.classifier(z))
        return x, _mean, _logvar 
    
    
class Generator(nn.Sequential):
    def __init__(self, img_size : tuple):
        super().__init__()
        self.z_dim = 256
        self.fc = nn.Linear(256, 256 * 4 * 4)
        n_c = (256, 128, 64, 32)
        blocks = []
        for i in range(3):
            blocks += [
                ResNetBlock(in_channels = n_c[i], out_channels = n_c[i+1], hid_channels = n_c[i+1], bias = True),
                nn.Upsample(scale_factor = 2)
            ]
        self.decoder = nn.Sequential(
            *blocks,
            ResNetBlock(in_channels = n_c[-1], out_channels = img_size[-1], hid_channels = img_size[-1], bias = True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels = img_size[-1], out_channels = img_size[1], kernel_size = 3, padding = 1),
            nn.Tanh(),
        )
        
        
    def forward(self, z):
        out = self.fc(z).view(z.shape[0], 256, 4, 4)
        return self.decoder(out)
    
