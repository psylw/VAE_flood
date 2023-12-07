
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [file for file in os.listdir(root_dir)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = np.load(img_name)[0:100,0:100]
        #image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return image
