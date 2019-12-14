"""
VAE Training
"""
import argparse
import os
import sys
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from tensorboardX import SummaryWriter
from data_loaders import SpriteTrainTest, TextureTrainTest
from modules import VAELoss, VAE, VAE_V2, LittleVAE, HierarchicalVAE, LittleVAE_V2, HierarchicalVAE_7x7, RegressionModel
from utils import get_dataset, create_little_vae
from tester import Tester

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

    device = torch.device(
        config["test"]["device"] if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # Build model architecture
    
    conditional = False
    model_params = config["test"]["model_params"]
    model_name = config["test"]["model"]
    if "hierarchical" in model_name:
        little_vae_params = config["test"]["little_vae_params"]
        little_vae = create_little_vae(**little_vae_params)
        little_vae.to(device)
    else:
        little_vae = None 
    model = MODELS[model_name](**model_params)  
    model.to(device)
    print("Model name: {}".format(model_name))

    # optionally resume from checkpoint

    resume = config["test"]["model_path"]
    print("Loading checkpoint {}".format(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint["state_dict"])

    # Get dataset
    regression_training_data, test_dataset, train_on_textures = get_dataset(config)
    b_size = 10
    validationloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=b_size,
                                                   shuffle=True)
    regression_train = torch.utils.data.DataLoader(regression_training_data,
                                                   batch_size = b_size,
                                                   shuffle=True)
    print("batch size: {}".format(b_size))
    # Define loss function (criterion) and optimizer
    loss_function = config["test"]["loss_function"]

    if "loss_parameters" in config["test"]:
        loss_parameters = config["test"]["loss_parameters"]
    else:
        loss_parameters = None
    if loss_parameters is not None:
        criterion = LOSS_FUNCTIONS[loss_function](**loss_parameters)
    else:
        criterion = LOSS_FUNCTIONS[loss_function]()


    kernel_size = config["test"]["kernel_size"]
    padding = config["test"]["padding"]
    stride = config["test"]["stride"]
    experiment_name = config["test"]["model_path"].split('/')[1]
#    reg_model = RegressionModel(16, 10, 3)
#    reg_model.to(device) 
    tester = Tester(device, model_name, experiment_name, little_vae, kernel_size, stride, padding,
 validationloader, model, loss_function, criterion, train_on_textures)
#    reg_model = tester.train_regression_model(reg_model, regression_train)
    tester.test()

if __name__ == '__main__':
    main()
