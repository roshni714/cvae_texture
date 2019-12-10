import torch.nn as nn
import torch
from .little_vae import LittleVAE
import torch.nn.functional as F
import math

class CenterCrop(nn.Module):
    def forward(self, inputs, th=64, tw=64):
        _, _,  w, h = inputs.shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return inputs[:, :, x1:x1+tw,y1:y1+th]


class HierarchicalVAE_7x7(nn.Module):
    def __init__(self, in_shape, n_latent):
        super(HierarchicalVAE_7x7, self).__init__()
        self.n_latent = n_latent
        conv_output_dim = 5184
        self.encoder_cnn = nn.Sequential(nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=2),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=2),
                                         nn.ReLU(),
                                         nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2),
                                         nn.ReLU()) 
        self.encoder_mlp = nn.Sequential(
	    nn.Linear(conv_output_dim, 64),
     	    nn.ReLU(),
	    nn.Linear(64, 2 * self.n_latent))
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.n_latent, 64*16),
            nn.ReLU(),
            nn.Linear(64*16, 16*64*64),
            nn.ReLU(),
        )

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        latent  = self.decode(z)
        return latent, mean, logvar

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = torch.randn(stddev.size()).to(stddev.device)
        return (noise * stddev) + mean

       
    def decode(self, z):
        out = self.decoder_mlp(z)
        out = out.view(z.size(0), 16, 64, 64)
        return out
 
    def encode(self, input_rep):
        res = self.encoder_cnn(input_rep) #padding + inputshape
        res  = self.encoder_mlp(res.view(res.size(0), -1))
        mean = res[:, :self.n_latent]
        logvar = res[:, self.n_latent:]
        return mean, logvar
         
