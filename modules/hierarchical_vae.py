import torch.nn as nn
import torch
from .spatial_broadcast import VAE_V2 
from .little_vae_v2 import LittleVAE_V2
import torch.nn.functional as F
import math

def create_kernel(window_size= [5, 5], eps=1e-6):
        r"""Creates a binary kernel to extract the patches. If the window size
        is HxW will create a (H*W)xHxW kernel.
        """
        window_range = window_size[0] * window_size[1]
        kernel = torch.zeros(window_range, window_range) + eps
        for i in range(window_range):
            kernel[i, i] += 1.0
        return kernel.view(window_range, 1, window_size[0], window_size[1])

def extract_patches_inverse(out_shape, y):
    permuted_y = y.permute(0, 4, 1, 2, 3)
    output = output_tmp.view()
def extract_image_patches(x):
    batch_size, channels, height, width = x.shape
    kernel = create_kernel().repeat(channels, 1, 1, 1)
    kernel = kernel.to(x.device).to(x.dtype)
    output_tmp = F.conv2d(
            x,
            kernel,
            stride=3,
            padding=1,
            groups=channels)

        # reshape the output tensor
    output = output_tmp.view(
            batch_size, channels, 5, 5, -1)
    return output.permute(0, 4, 1, 2, 3)  # BxNxCxhxw

class CenterCrop(nn.Module):
    def forward(self, inputs, th=64, tw=64):
        _, _,  w, h = inputs.shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return inputs[:, :, x1:x1+tw,y1:y1+th]


class HierarchicalVAE(nn.Module):
    def __init__(self, in_shape, n_latent, little_vae):
        super(HierarchicalVAE, self).__init__()
        self.little_vae = LittleVAE_V2([3, 5, 5], 16)
        self.n_latent = n_latent
        checkpoint = torch.load(little_vae)["state_dict"]
        self.little_vae.load_state_dict(checkpoint)
        self.encoder_mlp = nn.Sequential(nn.Linear(7056, 4096),
                                         nn.ReLU(),
                                         nn.Linear(4096, 2*n_latent),
                                         nn.ReLU())
        self.decoder_mlp = nn.Sequential(nn.Linear(n_latent, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 4096),
                                        nn.ReLU(),
                                        nn.Linear(4096, 7056),
                                        nn.ReLU())

    def forward(self, x):
        patches = extract_image_patches(x)
        next_layer_input = torch.zeros(x.shape[0], patches.shape[1], 1, 16).to(x.device).to(x.dtype)
        
        for i in range(patches.shape[1]):
                x1, y1 = self.little_vae.encode(patches[:, i, :, :, :]) 
                assert(torch.sum(torch.isnan(x1))==0) 
                assert(torch.sum(torch.isnan(y1))==0) 
                mean = self.little_vae.sample_z(x1, y1)
                assert(torch.sum(torch.isnan(mean))==0) 
                next_layer_input[:, i, :, :] += mean.reshape(x.shape[0], 1, 16)
        mean, logvar = self.encode(next_layer_input)
        assert(torch.sum(torch.isnan(mean))==0) 
        z = self.sample_z(mean, logvar)
        assert(torch.sum(torch.isnan(z))==0) 
        latent = self.decode(z).view(x.shape[0], patches.shape[1], 1, 16)
        to_decode = torch.zeros(x.shape[0], 3, 65, 65).to(device=x.device).to(dtype=x.dtype)
        for i in range(13):
            for j in range(13):
                row = 5*i
                col = 5*j
                to_decode[:, :, row:row+5, col:col+5] += self.little_vae.decode(latent[:,13*i + j, :, :])
        img = to_decode[:, :, :-1, :-1]
        assert(torch.sum(torch.isnan(latent))==0) 
        assert(torch.sum(torch.isnan(next_layer_input))==0) 

        return latent, mean, logvar, next_layer_input, img

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = torch.randn(stddev.size()).to(stddev.device)
        return (noise * stddev) + mean

       
    def decode(self, z):
        out = self.decoder_mlp(z)
        return out
 
    def encode(self, x):
        x  = self.encoder_mlp(x.view(x.size(0), -1))
        mean = x[:, :self.n_latent]
        logvar = x[:, self.n_latent:]
        return mean, logvar
         
                

        
 
