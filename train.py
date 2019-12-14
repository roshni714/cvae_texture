"""
VAE Training
"""
import argparse
import os
import sys
import torch
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from data_loaders import SpriteTrainTest, TextureTrainTest
from modules import VAELoss, VAE, VAE_V2, LittleVAE, HierarchicalVAE, LittleVAE_V2, HierarchicalVAE_7x7
from trainer import Trainer
from utils import get_dataset, create_little_vae
DEFAULT_CONFIG = os.path.dirname(__file__) + "configs/vae.yaml"
LOSS_FUNCTIONS = {"vae_loss": VAELoss}
MODELS = {"little_vae": LittleVAE, "hierarchical_vae": HierarchicalVAE, "vae_v2": VAE_V2, "little_vae_v2": LittleVAE_V2, "hierarchical_vae_7x7": HierarchicalVAE_7x7}

def main():
    """ Loads arguments and starts training."""
    parser = argparse.ArgumentParser(description="Deep Orientation Estimation")
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str)

    args = parser.parse_args()
    config_file = args.config

    # Load config
    assert os.path.exists(args.config), "Config file {} does not exist".format(
        args.config)

    with open(config_file) as fp:
        config = yaml.load(fp)

    if not os.path.exists(config["train"]["save_dir"]):
        os.makedirs(config["train"]["save_dir"])

    device = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Build model architecture
    
    conditional = False
    model_params = config["train"]["model_params"]
    model_name = config["train"]["model"]
    if "hierarchical" in model_name:
        little_vae_params = config["train"]["little_vae_params"]
        little_vae = create_little_vae(**little_vae_params)
        little_vae.to(device)
    else:
        little_vae = None 
    model = MODELS[model_name](**model_params)  
    model.to(device)
    print("Model name: {}".format(model_name))

    # optionally resume from checkpoint

    resume = config["train"]["resume"]
    if resume:
        if os.path.isfile(resume):
            print("Loading checkpoint {}".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint["epoch"]

            model.load_state_dict(checkpoint["state_dict"])

        else:
            start_epoch = 0
            print("No checkpoint found at {}".format(resume))

    else:
        start_epoch = 0

    # Get dataset
    train_dataset, test_dataset, train_on_textures = get_dataset(config)
    b_size = config["train"]["batch_size"] or 4
    validationloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=b_size,
                                                   shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size,
                                              shuffle=True)

    print("batch size: {}".format(b_size))
    # Define loss function (criterion) and optimizer
    learning_rate = config["train"]["learning_rate"] or 0.0001
    loss_function = config["train"]["loss_function"]

    if "loss_parameters" in config["train"]:
        loss_parameters = config["train"]["loss_parameters"]
    else:
        loss_parameters = None

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Set up tensorboard writer

    writer_train = SummaryWriter(
        "vae_runs/{}/training".format(config["train"]["save_as"]))

    # Train the network
    num_epochs = config["train"]["num_epochs"] or 2
    print("Number of epochs: {}".format(num_epochs))
    if loss_parameters is not None:
        criterion = LOSS_FUNCTIONS[loss_function](**loss_parameters)
    else:
        criterion = LOSS_FUNCTIONS[loss_function]()

    kernel_size = config["train"]["kernel_size"]
    padding = config["train"]["padding"]
    stride = config["train"]["stride"]
    experiment_name = config["train"]["save_as"]

    trainer = Trainer(device, model_name, experiment_name, little_vae, kernel_size, stride, padding)
    for epoch in range(start_epoch, num_epochs):
        trainer.train_epoch(
            trainloader, model, loss_function, criterion, optimizer,
            epoch, writer_train, train_on_textures)
        save_checkpoint(
            {'epoch': epoch + 1, 'state_dict': model.state_dict()},
            filename=os.path.join(config["train"]["save_dir"],
                                  'checkpoint_{}.tar'.format(epoch))
        )

    print('Finished training')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()
