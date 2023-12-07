# %%
import matplotlib.pyplot as plt
import torch
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import CustomImageFolder
import torch
import torch.nn as nn
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# %%
data_dir_train = 'sar\\noflood1'
data_dir_thresh = 'sar\\flood1'

filenames_sar1 = glob.glob('sar//noflood1//'+'*.npy')
filenames_sar2 = glob.glob('sar//flood1//'+'*.npy')

filenames_sar1.extend(filenames_sar2)
all_values=[]
for i in filenames_sar1:
    all_values.append(np.load(i).flatten())

std = np.std(all_values)
mean = np.mean(all_values)

# %%
transform = transforms.Compose([
    #transforms.Resize((64, 64)),
    
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std),

    # Add more transformations if required
])

# Create a custom dataset for the images without class folders
custom_dataset_train = CustomImageFolder(root_dir=data_dir_train, transform=transform)
custom_dataset_test = CustomImageFolder(root_dir=data_dir_thresh,transform=transform)

batch_size = 16
train_loader = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True)
data_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=False)
# %%
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

from test2 import VAE
vae = VAE().to(DEVICE)
# location 1

vae.load_state_dict(torch.load('model_weights1.pth'))


# Set the model to evaluation mode
vae.eval()

# Get a batch of data from the DataLoader
for images in data_loader:
    images = images.to(DEVICE)
    # Pass the batch through the VAE to get reconstructed outputs
    with torch.no_grad():
        
        reconstructed_images, _, _ = vae(images.float())
    
    # Plot original images and their reconstructions
    num_images = min(images.size(0), 5)  # Choose the number of images to visualize

    plt.figure(figsize=(10, 4))
    images = images.cpu()
    reconstructed_images=reconstructed_images.cpu()
    for i in range(num_images):
        # Original Images
        plt.subplot(2, 1, 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title('Original')
        plt.axis('off')

        # Reconstructed Images
        plt.subplot(2, 1, 2)
        plt.imshow(reconstructed_images[i].squeeze(), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

      # Stop after visualizing one batch of images
