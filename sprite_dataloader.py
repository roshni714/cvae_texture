import os
from skimage import io
from torch.utils.data import Dataset
from PIL import Image

class SpriteTrainTest():

    def __init__(self, config, image_transforms):
        train_path = config["train_path"]
        val_path = config["val_path"]
        self.train = SpriteDataloader(train_path, image_transforms)
        self.val = SpriteDataloader(val_path, image_transforms)

class SpriteDataloader(Dataset):
    """
    Stores a subset of the Drive Ahead dataset for which a data frame has
    already been created.
    Parameters:
        frame: a csv file that maps image file names to quaternion poses
        dataset_path: the directory where the images are stored
        photo_type:  "ir/" or "depth/" depending on whether IR or depth images
            are of interest
    """

    def __init__(self, dataset_path, image_transform=None):
        self.image_transform = image_transform
        self.dataset_path = dataset_path
    def __len__(self):
        return len(os.listdir(self.dataset_path))

    def __getitem__(self, idx):
        img_name = self.dataset_path + "/img_{}.png".format(idx)
        image = Image.fromarray(io.imread(img_name))
        if self.image_transform:
            image = self.image_transform(image)

        sample = {'image': image}

        return sample


