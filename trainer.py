import torch
from utils import AverageMeter


class Trainer(object):
    """ Trainer for Bingham Orientation Uncertainty estimation.
    Arguments:
        device (torch.device): The device on which the training will happen.
    """
    def __init__(self, device, floating_point_type="float"):
        self._device = device

    @staticmethod
    def adjust_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2

    def train_epoch(self, train_loader, model, loss_function, criterion,
                    optimizer, epoch, writer_train, writer_val, val_loader,
                    dataset_name):
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
            writer_val: A Tensorboard summary writer for reporting the average
                loss during validation.
            val_loader: A DataLoader that contains the shuffled validation set
            dataset_name: -
        """
        losses = AverageMeter()
        model.train()

        for i, data in enumerate(train_loader):
            if i % 20 == 0:
                self.validate(self._device, val_loader, model, loss_function,
                                 criterion, writer_val, i, epoch,
                                 len(train_loader), 0.01, dataset_name)

                # switch to train mode
            model.train()
            input_var = data["image"].float().to(self._device)

            output, mean, logvar = model(input_var)
            loss = criterion(output, input_var, mean, logvar, epoch)
             # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), data["image"].size(0))
            writer_train.add_scalar('data/loss', loss,
                                    i + len(train_loader) * epoch)

            print("Epoch: [{0}][{1}/{2}]\t Loss {loss.last_val:.4f} "
                  "({loss.avg:.4f})\t".format(
                    epoch, i, len(train_loader), loss=losses))


    def validate(self, device, val_loader, model, loss_function, criterion, writer,
                 index=None, cur_epoch=None, epoch_length=None, eval_fraction=1,
                 dataset_name="UPNAHeadPose"):
        """
        Method that validates the model on the validation set and reports losses
        to Tensorboard using the writer
        device: A string that states whether we are using GPU ("cuda:0") or cpu
        model: The model we are training
        loss_function: String name of loss we are using (ie. "mse")
        optimizer: The optimizer we are using
        writer: A Tensorboard summary writer for reporting the average loss
            during validation.
        cur_epoch: integer epoch number representing the training epoch we are
            currently on.
        index: Refers to the batch number we are on within the training set
        epoch_length: The number of batches in an epoch
        val_loader: A DataLoader that contains the shuffled validation set
        loss_parameters: Parameters passed on to the loss generation class.
        """
        # switch to evaluate mode
        model.eval()

        losses = AverageMeter()
        val_load_iter = iter(val_loader)
        for i,data in enumerate(val_loader):
            input_var = data["image"].float().to(device)
                # compute output
            output, mean, logvar = model(input_var)
            loss = criterion(output, input_var, mean, logvar)
            # measure accuracy and record loss
            losses.update(loss.item(), data["image"].size(0))

        if index is not None:
            cur_iter = cur_epoch * epoch_length + index
            writer.add_scalar('data/loss', losses.avg, cur_iter)
            print(output[0].shape)
            writer.add_image('orignal_image', input_var[0], cur_iter)
            writer.add_image('output_image', output[0], cur_iter)
            print('Test:[{0}][{1}/{2}]\tLoss {loss.last_val:.4f} '
                  '({loss.avg:.4f})\t'.format(
                    cur_epoch, index, epoch_length, loss=losses))
