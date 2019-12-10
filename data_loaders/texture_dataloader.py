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
        crop_size = config["crop"]
        self.train = TextureDataloader(train_path, texture_path, crop_size, image_transforms)
        self.val = TextureDataloader(val_path, texture_path, crop_size, image_transforms)

class TextureDataloader(Dataset):
    """
    Simple Dataloader for Sprite masks multipled by texture dataset
    """
    def __init__(self, dataset_path, texture_path, crop_size, image_transform=None):
        self.image_transform = image_transform
        self.dataset_path = dataset_path
        self.texture_path = texture_path
        self.crop_size = crop_size
        self.random_crop = transforms.Compose([transforms.Resize((80, 80)),
                                               transforms.RandomCrop((64, 64))]) 
    def __len__(self):
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, idx):
        img_name = self.dataset_path + "/img_{}.png".format(idx)
        image = Image.fromarray(io.imread(img_name))

        texture_name = self.texture_path + "/img_{}.jpg".format(13)
        texture = self.random_crop(Image.fromarray(io.imread(texture_name))) 
       
        if self.image_transform:
            image = torch.round(self.image_transform(image)).float()
            cropped_texture = transforms.ToTensor()(transforms.RandomCrop((self.crop_size[0], self.crop_size[1]))(texture))
            texture = self.image_transform(texture)

        assert(torch.sum(torch.isnan(image))==0)
        assert(torch.sum(torch.isnan(texture))==0)

        after_mask = image * texture
        assert(torch.sum(torch.isnan(after_mask))==0)

        c, h, w = image.shape
        avg_color_pattern = [torch.mean(after_mask[i][torch.nonzero(image[i]).split(1, dim=1)]) for i in range(c)]
        assert(torch.sum(torch.isnan(torch.tensor(avg_color_pattern)))==0)

        mean_color= torch.ones(c, h, w)
        for i in range(c):
            mean_color[i] *= avg_color_pattern[i]
        assert(torch.sum(torch.isnan(mean_color))==0)
        final_image = torch.where(after_mask <= 1e-5, mean_color, after_mask)
        sample = {'image': final_image, 'texture': cropped_texture}
        assert(torch.sum(torch.isnan(final_image))==0)
        return sample


