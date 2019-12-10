import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

class CenterCrop(nn.Module):
    def forward(self, inputs, th=7, tw=7):
        _, _,  w, h = inputs.shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return inputs[:, :, x1:x1+tw,y1:y1+th]

class LittleVAE(nn.Module):
    def __init__(self, in_shape, n_latent):
        super(LittleVAE, self).__init__()
        self.in_shape = in_shape
        self.n_latent = n_latent
        c,h,w = in_shape
        print(in_shape)
        conv_output_dim = 64 * (h+1) * (h+1) # channels * height * width
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=int((h+1)/2), stride=1, padding=2),  # 32, 16, 16
            nn.ReLU(),
        )
        self.encoder_mlp = nn.Sequential(
	    nn.Linear(conv_output_dim, 64),
     	    nn.ReLU(),
	    nn.Linear(64, 2 * n_latent))
        self.decoder_mlp = nn.Sequential(
            nn.Linear(n_latent, 64),
            nn.ReLU(),
            nn.Linear(64, conv_output_dim),
            nn.ReLU()
        ) 
        self.decoder_cnn = nn.Sequential( 
            nn.ConvTranspose2d(64, c, kernel_size=int(h+1/2), stride=1, padding=2),
            nn.ReLU(),
            CenterCrop(),
            nn.Sigmoid()
        )

    def sample_z(self, mean, logvar):

        stddev = torch.exp(0.5 * logvar)
        noise = torch.randn(stddev.size()).to(stddev.device)
        return (noise * stddev) + mean

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_mlp(x)
        mean = x[:, :self.n_latent]
        logvar = x[:, self.n_latent:]
        return mean, logvar

    def decode(self, z):
        out = self.decoder_mlp(z)
        out = out.view(z.size(0), 64, self.in_shape[1]+1, self.in_shape[1]+1)
        out = self.decoder_cnn(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar


