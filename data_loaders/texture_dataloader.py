import torch
import os
from skimage import io
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms as transforms

class TextureTrainTest():
    def __init__(self, config, image_transforms):
        train_path = config["train_path"]
        val_path = config["val_path"]
        texture_path = config["texture_path"]
        self.train = TextureDataloader(train_path, texture_path, image_transforms)
        self.val = TextureDataloader(val_path, texture_path, image_transforms)

class TextureDataloader(Dataset):
    """
    Simple Dataloader for Sprite masks multipled by texture dataset
    """
    def __init__(self, dataset_path, texture_path, image_transform=None):
        self.image_transform = image_transform
        self.dataset_path = dataset_path
        self.texture_path = texture_path

        self.random_crop = transforms.Compose([transforms.Resize((64, 64)), 
                                              transforms.RandomRotation(180)])
    def __len__(self):
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, idx):
        img_name = self.dataset_path + "/img_{}.png".format(idx)
        image = Image.fromarray(io.imread(img_name))

        texture_name = self.texture_path + "/img_{}.jpg".format(2)
        texture = self.random_crop(Image.fromarray(io.imread(texture_name))) 
       
        if self.image_transform:
            image = torch.round(self.image_transform(image)).float()
            cropped_texture = transforms.ToTensor()(transforms.RandomCrop((5, 5))(texture))
            texture = self.image_transform(texture)

        c, h, w = image.shape
        avg_color_pattern = [torch.mean(texture[i]) for i in range(c)]
        mean_color= torch.ones(c, h, w)
        for i in range(c):
            mean_color[i] *= avg_color_pattern[i]

        after_mask = image * texture
        final_image = torch.where(image <= 1e-5, mean_color, after_mask)
        
        sample = {'image': final_image, 'texture': cropped_texture}

        return sample


