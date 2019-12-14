import torch
from utils import AverageMeter, extract_image_patches
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
from modules import RegressionModel
class Tester(object):
    def __init__(self, device, model_name, experiment_name, little_vae, kernel_size, stride, padding,                 dataset, model, loss_function, criterion, use_textures):
        self._device = device
        self._experiment_name = experiment_name
        self._model_name = model_name
        self._little_vae = little_vae
        self._padding = padding
        self._kernel_size = kernel_size
        self._stride = stride
        self._dataset = dataset
        self._model = model
        self._model.eval()
        self._loss_function = loss_function
        self._criterion = criterion
        self._use_textures = use_textures
        self._decoding_function = make_decoding_function(little_vae=self._little_vae, kernel_size=self._kernel_size, padding=self._padding)

    def generate_little_vae_encoding(self, input_var):
        b, c, h, w = input_var.shape
        patches = extract_image_patches(input_var, self._kernel_size, self._stride, self._padding)
        next_layer_input = torch.zeros(input_var.shape[0], self._little_vae.n_latent, h+self._padding, w+self._padding).to(input_var.device).to(input_var.dtype)
        division = torch.zeros(input_var.shape[0], self._little_vae.n_latent, h+self._padding, w+self._padding).to(input_var.device).to(input_var.dtype) #inputshape + kernel size
        for i in range(h):
            row = i+ self._padding #i + padding
            for j in range(w):
                col = j + self._padding #1 + padding
                mean, _ = self._little_vae.encode(patches[:, i*h+j, :, :, :])
                next_layer_input[:, :, row, col] += mean
        input_rep = next_layer_input[:, :, self._padding: self._padding + h, self._padding: self._padding + h]
        return input_rep


    def generate_little_vae_decoding(self, latent_rep):
        return self._decoding_function(latent_rep)

    def generate_plotted_results(self, batched_img_list, cur_iter):

        n_rows = len(batched_img_list)
        n_cols = batched_img_list[0].shape[0]
        all_ims = []

        if n_rows == 2:
            titles = ["Input Image", "VAE Recons."]
        elif n_rows == 3:
            titles = ["Input Image", "HVAE Recons.", "Texture-VAE Recons."]

        for i in range(n_rows):
            for j in range(n_cols):
                res = batched_img_list[i][j].detach().cpu().numpy()
                img = np.transpose(res, (1, 2, 0))
                all_ims.append(img)

        fig = plt.figure(figsize=(n_cols, n_rows))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                 axes_pad=0.01,  # pad between axes in inch.
                 )
        for i in range(n_rows):
            grid[i*n_cols].set_ylabel(titles[i], fontsize=6)

        for ax, im in zip(grid, all_ims):
            ax.imshow(im)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.get_xaxis().set_visible(False) 
            ax.tick_params(axis='both', which='both', length=0)
        plt.savefig("recons/{}/{}.pdf".format(self._experiment_name, cur_iter))
        plt.close('all')
        print(cur_iter)


    def test(self):
        """
            Method that trains the model for one epoch on the training set and
            reports losses to Tensorboard using the writer_train
            train_loader: A DataLoader that contains the shuffled training set
            model: The model we are training
            loss_function: String name of loss we are using (ie. "mse")
            optimizer: The optimizer we are using
            Epoch: integer epoch number
            writer_train: A Tensorboard summary writer for reporting the average
                loss while training.
            val_loader: A DataLoader that contains the shuffled validation set
            dataset_name: -
        """
#        reg_model = RegressionModel(16, 10, 3)
#        checkpoint = torch.load("reg_model.tar")
#        reg_model.load_state_dict(checkpoint["state_dict"])
#        reg_model.to(self._device)
        losses = AverageMeter()
#        position_regression = AverageMeter()
#        size_regression = AverageMeter()
#        mse = nn.MSELoss(reduction="sum")
        if self._use_textures:
            input_name = "texture"
        else:
            input_name = "image"
        self._model.eval()
        for i, data in enumerate(self._dataset):
               # switch to train mode
            input_var = data[input_name].float().to(self._device)
            if "hierarchical" in self._model_name:
                input_rep = self.generate_little_vae_encoding(input_var)
                latent_rep, mean, logvar = self._model(input_rep)
                output = self.generate_little_vae_decoding(latent_rep)
                loss = self._criterion(latent_rep, input_rep, mean, logvar)
            else:
                output, mean, logvar =self._model(input_var)
                loss = self._criterion(output, input_var, mean, logvar)

 #           true_position = data["features"].float().to(self._device)
 #           predicted_position = reg_model(mean)
 #           pos_error = mse(true_position[0:1]*64, predicted_position[0:1]*64)/input_var.shape[0]
 #           size_error = mse(true_position[2]*100, predicted_position[2]*100)/input_var.shape[0]
 #           size_regression.update(size_error.item(), 1)
 #           position_regression.update(pos_error.item(), 1)
            loss["loss"]/=input_var.shape[0]
            losses.update(loss["loss"].item(), 1)
           
            cur_iter = i
            input_img = input_var
            print(cur_iter)

            if "hierarchical" in self._model_name:
                hierarchical_output = output
                little_vae_output = self.generate_little_vae_decoding(input_rep)
                self.generate_plotted_results([input_img, hierarchical_output, little_vae_output], cur_iter)
            else:
                output_img =  output
                self.generate_plotted_results([input_img, output_img], cur_iter)         

            if "hierarchical" in self._model_name:
                self._model.traverse(input_rep, self._experiment_name,self._decoding_function , cur_iter)
            else:
                self._model.traverse(input_var, self._experiment_name, cur_iter)
 
#        print("Position Error: {}, Size Error: {}".format(position_regression.avg, size_regression.avg))

    def train_regression_model(self, reg_model, regression_training_data):
        self._model.eval()
        reg_model.train()
        optimizer = optim.SGD(reg_model.parameters(), lr=0.001) 
        mse = nn.MSELoss(reduction="sum")
        losses = AverageMeter()
        for epoch in range(1):
            for i, data in enumerate(regression_training_data):
                input_var = data["image"]
                target_position = data["features"].float().to(self._device)
                if "hierarchical" in self._model_name:
                    input_rep = self.generate_little_vae_encoding(input_var)
                    latent_rep, mean, logvar = self._model(input_rep.to(self._device))
                else:
                    output, mean, logvar = self._model(input_var)

                predicted_position=reg_model(mean)
                loss = mse(target_position, predicted_position)
                optimizer.zero_grad()
                loss.backward()
                losses.update(loss.item()/input_var.shape[0], 1)
                optimizer.step()
                print("Epoch: [{0}][{1}/{2}]\t Loss {loss.last_val:.4f} "
                  "({loss.avg:.4f})\t".format(
                    epoch, i, len(regression_training_data), loss=losses))
         
        torch.save({'state_dict': reg_model.state_dict()}, 'reg_model.tar') 
 
def make_decoding_function(little_vae, kernel_size, padding):
     def decoding_function(latent_rep):
         z = latent_rep
         b, c, h, w = z.shape
         division = torch.zeros(b, 3, h + kernel_size[0], w+ kernel_size[1])
         to_decode = torch.zeros(b,3, h + kernel_size[0], w+ kernel_size[1])
         for i in range(64):
             row = i
             for j in range(64):
                 col = j
                 recons = little_vae.decode(latent_rep[:,:, i, j]).detach().cpu()
                 to_decode[:, :, row:row+kernel_size[0], col:col+kernel_size[1]] += recons
                 division[:, :, row:row+kernel_size[0], col:col+kernel_size[1]] += torch.ones(recons.shape)
         to_decode /= division
         img = to_decode[:, :, padding: h + padding, padding: w + padding]
         return img
     return decoding_function

   
   
