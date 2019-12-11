import sys
import torch
from modules import LittleVAE
from data_loaders import SpriteTrainTest, TextureTrainTest
import torch.nn.functional as F
import torchvision.transforms as transforms


def create_kernel(window_size= [7, 7], eps=1e-6):
        r"""Creates a binary kernel to extract the patches. If the window size
        is HxW will create a (H*W)xHxW kernel.
        """
        window_range = window_size[0] * window_size[1]
        kernel = torch.zeros(window_range, window_range) + eps
        for i in range(window_range):
            kernel[i, i] += 1.0
        return kernel.view(window_range, 1, window_size[0], window_size[1])

def extract_image_patches(x, kernel_size, stride, padding):
    batch_size, channels, height, width = x.shape
    kernel = create_kernel(kernel_size).repeat(channels, 1, 1, 1)
    kernel = kernel.to(x.device).to(x.dtype)
    output_tmp = F.conv2d(
            x,
            kernel,
            stride=stride,
            padding=padding,
            groups=channels)

        # reshape the output tensor
    output = output_tmp.view(
            batch_size, channels, kernel_size[0], kernel_size[1], -1)
    return output.permute(0, 4, 1, 2, 3)  # BxNxCxhxw



def get_dataset(config):
    """ Returns the training data using the provided configuration."""

    data_loader = config["data_loader"]
    size = data_loader["input_size"]
    
    image_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])

    train_on_textures = False
    if data_loader["name"] == "Sprite":
        dataset = SpriteTrainTest(config['data_loader'], image_transforms)
        train_dataset = dataset.train
        val_dataset = dataset.val
    elif data_loader["name"] == "Texture":
        dataset = TextureTrainTest(config['data_loader'], image_transforms)
        train_dataset = dataset.train
        val_dataset = dataset.val
        train_on_textures = data_loader["train_on_textures"]
    else:
        sys.exit("Unknown data loader " + config['data_loader']["name"] + ".")

    return train_dataset, val_dataset, train_on_textures

def create_little_vae(in_shape, n_latent, path):
        little_vae = LittleVAE(in_shape, n_latent)
        checkpoint = torch.load(path)["state_dict"]
        little_vae.load_state_dict(checkpoint)
        little_vae.eval()
        return little_vae
 


#two methods from Nicholas Watters
def images_to_grid(images,
                   grid_height=4,
                   grid_width=4,
                   image_border_value=0.5):
    """Combine images and arrange them in a grid.

    Args:
        images: Tensor of shape [B, H], [B, H, W], or [B, H, W, C].
        grid_height: Height of the grid of images to output, or None. Either
            `grid_width` or `grid_height` must be set to an integer value. If None,
            `grid_height` is set to ceil(B/`grid_width`), and capped at
            `max_grid_height` when provided.
        grid_width: Width of the grid of images to output, or None. Either
            `grid_width` or `grid_height` must be set to an integer value. If None,
            `grid_width` is set to ceil(B/`grid_height`), and capped at
            `max_grid_width` when provided.
        max_grid_height: Maximum allowable height of the grid of images to output,
            or None. Only used when `grid_height` is None.
        max_grid_width: Maximum allowable width of the grid of images to output,
            or None. Only used when `grid_width` is None.
        image_border_value: None or scalar value of greyscale borderfor images. If
            None, then no border is rendered.

    Raises:
        ValueError: if neither of grid_width or grid_height are set to a positive
            integer.

    Returns:
        images: Tensor of shape [height*H, width*W, C]. C will be set to 1 if the
        input was provided with no channels. Contains all input images in a grid.
    """

    # If only one dimension is set, infer how big the other one should be.
    images = images[:grid_height * grid_width, ...]

    # Pad with extra blank frames if grid_height x grid_width is less than the
    # number of frames provided.
    pre_images_shape = images.get_shape().as_list()
    if pre_images_shape[0] < grid_height * grid_width:
        pre_images_shape[0] = grid_height * grid_width - pre_images_shape[0]
        if image_border_value is not None:
            dummy_frames = image_border_value * torch.ones(shape=pre_images_shape,
                                                        dtype=images.dtype)
        else:
            dummy_frames = torch.zeros(shape=pre_images_shape, dtype=images.dtype)
            images = torch.concat([images, dummy_frames], axis=0)

    if image_border_value is not None:
        images = _pad_images(images, image_border_value=image_border_value)
    images_shape = images.get_shape().as_list()
    images = torch.reshape(images, [grid_height, grid_width] + images_shape[1:])
    if len(images_shape) == 2:
        images = torch.expand_dims(images, -1)
    if len(images_shape) <= 3:
        images = torch.expand_dims(images, -1)
    image_height, image_width, channels = images.get_shape().as_list()[2:]
    images = torch.transpose(images, perm=[0, 2, 1, 3, 4])
    images = torch.reshape(
        images, [grid_height * image_height, grid_width * image_width, channels])
    return images


def convert_to_tensorboard_image_shape(image):
  """Ensures an image has shape [B, H, W, C], used for visualizing latents."""
  image_shape = image.get_shape().as_list()
  if image_shape[-1] not in [1, 3]:
    image = torch.expand_dims(image, -1)
    image_shape = image.get_shape().as_list()
  output_image_shape = (4 - len(image_shape)) * [1] + image_shape
  if output_image_shape[0] is None:
    output_image_shape[0] = -1

  image = torch.reshape(image, output_image_shape)
  return image


class AverageMeter(object):
    """Computes and stores the averages over a numbers or dicts of numbers.
    For the dict, this class assumes that no new keys are added during
    the computation.
    """

    def __init__(self):
        self.last_val = 0
        self.avg = 0 
        self.count = 0 

    def update(self, val, n=1):
        self.last_val = val
        n = float(n)
        if type(val) == dict:
            if self.count == 0:
                self.avg = copy.deepcopy(val)
            else:
                for key in val:
                    self.avg[key] *= self.count / (self.count + n)
                    self.avg[key] += val[key] * n / (self.count + n)
        else:
            self.avg *= self.count / (self.count + n)
            self.avg += val * n / (self.count + n)

        self.count += n
        self.last_val = val
