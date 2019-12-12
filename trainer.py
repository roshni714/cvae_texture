import torch
from utils import AverageMeter, extract_image_patches


class Trainer(object):
    """
    Arguments:
        device (torch.device): The device on which the training will happen.
    """
    def __init__(self, device, model_name, experiment_name, little_vae, kernel_size, stride, padding):
        self._device = device
        self._experiment_name = experiment_name
        self._model_name = model_name
        self._little_vae = little_vae
        self._padding = padding
        self._kernel_size = kernel_size
        self._stride = stride

    @staticmethod
    def adjust_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2

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
        z = latent_rep
        b, c, h, w = z.shape
        division = torch.zeros(b, 3, h + self._kernel_size[0], w+ self._kernel_size[1])
        to_decode = torch.zeros(b,3, h + self._kernel_size[0], w+ self._kernel_size[1])
        for i in range(64):
            row = i
            for j in range(64):
                col = j
                recons = self._little_vae.decode(latent_rep[:,:, i, j]).detach().cpu()
                to_decode[:, :, row:row+self._kernel_size[0], col:col+self._kernel_size[1]] += recons
                division[:, :, row:row+self._kernel_size[0], col:col+self._kernel_size[1]] += torch.ones(recons.shape)
        to_decode /= division
        img = to_decode[:, :, self._padding: h + self._padding, self._padding: w + self._padding]
        return img
 
    def train_epoch(self, train_loader, model, loss_function, criterion,
                    optimizer, epoch, writer_train, writer_val, val_loader,
                    train_on_textures):
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
        losses = AverageMeter()
        model.train()

        if train_on_textures:
            input_name = "texture"
        else:
            input_name = "image"
        for i, data in enumerate(train_loader):
               # switch to train mode
            model.train()
            input_var = data[input_name].float().to(self._device)
            if "hierarchical" in self._model_name:
                input_rep = self.generate_little_vae_encoding(input_var)
                latent_rep, mean, logvar = model(input_rep)
                output = self.generate_little_vae_decoding(latent_rep[0:1])
                loss = criterion(latent_rep, input_rep, mean, logvar, epoch)
            else:
                output, mean, logvar = model(input_var)
                loss = criterion(output, input_var, mean, logvar, epoch)

             # compute gradient and do optimization step
            optimizer.zero_grad()
            loss["loss"].backward()
            optimizer.step()
            loss["loss"]/=input_var.shape[0]
            losses.update(loss["loss"].item(), 1)
           
            cur_iter = i + len(train_loader) * epoch
            for key in loss: 
                writer_train.add_scalar('data/{}'.format(key), loss[key].item(), cur_iter)
            print("Epoch: [{0}][{1}/{2}]\t Loss {loss.last_val:.4f} "
                  "({loss.avg:.4f})\t".format(
                    epoch, i, len(train_loader), loss=losses))
            if i % 10 == 0:
                writer_train.add_image('imgs/input', input_var[0], cur_iter)
                if "hierarchical" in self._model_name:
                    writer_train.add_image('imgs/hierarchical_output', output[0], cur_iter)
                    writer_train.add_image('imgs/little_vae_output', self.generate_little_vae_decoding(input_rep[0:1])[0], cur_iter)
                else:
                    writer_train.add_image('imgs/output', output[0], cur_iter)

#            if i %100 == 0:
#                if "hierarchical" in self._model_name:
#                    model.traverse(input_var, self._experiment_name, self.little_vae, cur_iter)
#                else:
#                    model.traverse(input_var, self._experiment_name, cur_iter)
 
