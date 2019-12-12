import torch.nn as nn
import torch
from .little_vae import LittleVAE
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

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
            nn.Linear(64*16, 5184),
            nn.ReLU(),
        )
        self.decoder_cnn = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=2),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=2),
                                         nn.ReLU(),
                                         CenterCrop()) 

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
        out = out.view(z.size(0),64, 9, 9)
        out = self.decoder_cnn(out)
        return out
 
    def encode(self, input_rep):
        print(torch.mean(input_rep))
        res = self.encoder_cnn(input_rep) #padding + inputshape
        res  = self.encoder_mlp(res.view(res.size(0), -1))
        mean = res[:, :self.n_latent]
        logvar = res[:, self.n_latent:]
        return mean, logvar

    def traverse(self, image, name, decoding_function, cur_iter, figure_width=10.5, num_cols=9, image_height=1.5):
        """
            Plot a traversal of the latent space.
    
            Steps are:
               1) encode an input to a latent representation
               2) adjust each latent value from -3 to 3 while keeping other values fixed
               3) decode each adjusted latent representation
               4) display
        """
        mu, logvar = self.encode(image)
        z = mu  # since we're not training, no noise is added
    
        num_rows = z.shape[-1]
        num_cols = num_cols
    
        fig = plt.figure(figsize=(figure_width, image_height * num_rows))
    
        for i in range(num_rows):
            z_i_values = np.linspace(-3.0, 3.0, num_cols)
            z_i = z[0][i].detach().cpu().numpy()
            z_diffs = np.abs((z_i_values - z_i))
            j_min = np.argmin(z_diffs)
            for j in range(num_cols):
                z_i_value = z_i_values[j]
                if j != j_min:
                     z[0][i] = z_i_value
                else:
                     z[0][i] = float(z_i)
                
                x = decoding_function(self.decode(z)).detach().cpu().numpy()
            
                ax = fig.add_subplot(num_rows, num_cols, i * num_cols + j + 1)
                ax.imshow(x[0].transpose())
            
                if i == 0 or j == j_min:
                    ax.set_title('{}'.format(round(z[0][i]), 2))
            
                if j == j_min:
                    ax.set_xticks([], [])
                    ax.set_yticks([], []) 
                    color = 'mediumseagreen'
                    width = 8
                    for side in ['top', 'bottom', 'left', 'right']:
                        ax.spines[side].set_color(color)
                        ax.spines[side].set_linewidth(width)
                else:
                    ax.axis('off')
            z[0][i] = float(z_i)
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.04)
        plt.savefig("traversals/{}/traverse_{}.png".format(name, cur_iter))               
