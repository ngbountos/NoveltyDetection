import torch
from torch import nn

import numpy as np
from torch.nn.utils import spectral_norm


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 6, kernel_size=4)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(6,12,kernel_size=4)),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2,return_indices=True)
        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(12, 16, kernel_size=4)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(16,16,kernel_size=4)),
            nn.ReLU()
        )
        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(16,16,kernel_size=4)),
            nn.ReLU(),
            spectral_norm(nn.ConvTranspose2d(16,12,kernel_size=4)),
            nn.ReLU())
        self.unpool = nn.MaxUnpool2d(2)
        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(12,6,kernel_size=4)),
            nn.ReLU(),
            spectral_norm(nn.ConvTranspose2d(6,3,kernel_size=4)),
            nn.ReLU())

    def forward(self, x):

        x = self.encoder_1(x)
        x , indices = self.pool(x)
        x = self.encoder_2(x)
        x, indices2 = self.pool(x)
        z = x
        x = self.unpool(x,indices=indices2)
        x = self.decoder_2(x)
        x = self.unpool(x, indices= indices)
        x = self.decoder_1(x)

        return x



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

        self.Generator = AE() #nn.Sequential(self.encoder,self.decoder)

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
            nn.Linear(242,1),
            nn.Sigmoid()
            )

        self.noise_std = noise_std
        self.device = device

    def forward(self,x):
        noise = np.random.normal(0, self.noise_std, size=x.shape)
        noise = torch.from_numpy(noise).float().to(self.device)
        x_noise = (x + noise).clamp(0.0, 1.0)
        x_recon = self.Generator(x_noise)
        y = self.Discriminator(x_recon)
        #y = torch.nn.Sigmoid(y)
        return y




class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(3,6,kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6,16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16,32, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32,64,kernel_size=5)
        self.bn4 = nn.BatchNorm2d(64)


        self.linear = nn.Linear(1465472,1000)
        self.final = nn.Linear(1000,2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = nn.Flatten()(x)
        x = self.linear(x)
        x = self.final(x)

        return x
