import torch
from torch.nn import functional as F

class VAELoss():
    def __init__(self):
        self.name = "vae_loss"
        self.mse = torch.nn.MSELoss(reduction='sum')
        
#        self.cur_epoch = 0
#        self.beta = 1
    def __call__(self, output, input_var, mean, logvar, epoch=None):
        loss = 0
        recon_loss = 0
        kl_loss = 0
        dim_prod = output.shape[1] * output.shape[2] * output.shape[3]
#        if self.beta < 1 and epoch > self.cur_epoch:
#            self.beta = min(1, self.beta *1.1) 
#            self.cur_epoch = max(self.cur_epoch, epoch)

        for i in range(len(output)):
#            recon_loss = F.binary_cross_entropy(output[i].view(-1, dim_prod), input_var[i].view(-1, dim_prod), reduction='sum')
            recon_loss += self.mse(output[i], input_var[i])
            kl_loss +=0.5 * torch.sum(torch.exp(logvar[i]) + mean[i]**2 - 1. - logvar[i])
        loss = recon_loss + kl_loss

        return {"loss": loss, "recon_loss": recon_loss/output.shape[0], "kl_loss": kl_loss/output.shape[0]}
