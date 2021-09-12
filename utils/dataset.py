from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image


class HumanDataset(Dataset):
    def __init__(self, image_path, mask_path, augmentations=None):
        self.dir_image_path = image_path
        self.dir_mask_path = mask_path

        self.images = os.listdir(image_path)
        self.masks = os.listdir(mask_path)

        self.augmentations = augmentations
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        image_path = os.path.join(self.dir_image_path, self.images[idx])
        mask_path = os.path.join(self.dir_mask_path, self.masks[idx])

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

        if self.augmentations is not None:
            image, mask = self.augmentations(image=image, mask=mask).values()

        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        return image, mask

    def __len__(self):
        return len(self.images)
