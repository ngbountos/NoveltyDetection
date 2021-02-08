import torch
from torch import nn

import numpy as np


class NoveltyDetector(nn.Module):

    def __init__(self, noise_std: float, device='cpu'):
        super(NoveltyDetector,self).__init__()

        #Generator
        #Input 226x226
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16 , stride=2,kernel_size=5), #111x111
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=5,stride=2), #54x54
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=5,stride=2), #25x25
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5,stride=2), #11x11
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), #25x25
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), #53x53
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16,kernel_size=7, stride=2), #111x111
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=6, stride=2),#226x226
            nn.Sigmoid()
            )

        self.Generator = nn.Sequential(self.encoder,self.decoder)

        #Discriminatior
        self.Discriminator = nn.Sequential(
            nn.Conv2d(3, 16,kernel_size=5 ,stride = 2), #112x112
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=5,stride=2), #55x55
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,64, kernel_size=5, stride=2), #27x27
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=5,stride=2), #13x13
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(242,2)
            )

        self.noise_std = noise_std
        self.device = device

    def forward(self,x):
        noise = np.random.normal(0, self.noise_std, size=x.shape)
        noise = torch.from_numpy(noise).float().to(self.device)
        x_noise = (x + noise).clamp(0.0, 1.0)
        x_recon = self.Generator(x_noise)
        y = self.Discriminator(x_recon)
        y = torch.sigmoid(y)
        return y