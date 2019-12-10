import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

class CenterCrop(nn.Module):
    def forward(self, inputs, th=5, tw=5):
        _, _,  w, h = inputs.shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return inputs[:, :, x1:x1+tw,y1:y1+th]

class LittleVAE_V2(nn.Module):
    def __init__(self, in_shape, n_latent):
        super(LittleVAE_V2, self).__init__()
        self.in_shape = in_shape
        self.n_latent = n_latent
        c,h,w = in_shape
        conv_output_dim = 64 * (h+2) * (h+2) # channels * height * width
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=int((h+1)/2), stride=1, padding=2),  # 32, 16, 16
            nn.ReLU(),
        )
        self.encoder_mlp = nn.Sequential(
	    nn.Linear(conv_output_dim, 256),
     	    nn.ReLU(),
	    nn.Linear(256, 64),
            nn.ReLU(),
	    nn.Linear(64, 2 * n_latent),
            nn.ReLU())
        self.decoder_mlp = nn.Sequential(
            nn.Linear(n_latent, 64),
            nn.ReLU(),
	    nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, conv_output_dim),
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

#       noise = torch.tensor(np.random.normal(0, 0.3, (mean.shape[0], 16))).to(device=mean.device).to(dtype=mean.dtype)
#        return torch.sigmoid(mean) + noise

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_mlp(x)
        mean = x[:, :self.n_latent]
        logvar = x[:, self.n_latent:]
        return mean, logvar

    def decode(self, z):
        out = self.decoder_mlp(z)
        out = out.view(z.size(0), 64, 7,7)
        out = self.decoder_cnn(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar



