import torch
from torch.nn import functional as F

class VAELoss():
    def __init__(self):
        self.name = "vae_loss"
        self.mse = torch.nn.MSELoss(reduction='sum')
        
    def __call__(self, output, input_var, mean, logvar, epoch=None):
        loss = 0
        recon_loss = 0
        kl_loss = 0

        for i in range(len(output)):
            recon_loss += self.mse(output[i], input_var[i])
            kl_loss +=0.5 * torch.sum(torch.exp(logvar[i]) + mean[i]**2 - 1. - logvar[i])
        
        loss = recon_loss + kl_loss

        return {"loss": loss, "recon_loss": recon_loss/output.shape[0], "kl_loss": kl_loss/output.shape[0]}
