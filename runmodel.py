# %%
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataloader import CustomImageFolder
import torch
import torch.nn as nn

import glob
import numpy as np
# %%
data_dir_train = 'sar\\noflood1'
data_dir_thresh = 'sar\\flood1'

filenames_sar1 = glob.glob('sar//noflood1//'+'*.npy')
filenames_sar2 = glob.glob('sar//flood1//'+'*.npy')

filenames_sar1.extend(filenames_sar2)
all_values=[]
for i in filenames_sar1:
    all_values.extend(np.load(i).flatten())

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
# Create a DataLoader for the custom dataset
# Adjust batch_size, shuffle, and other parameters as needed
batch_size = 16
train_loader = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=False)
# %%
'''
# Iterate through the DataLoader for unsupervised learning or other purposes
for batch_idx, data in enumerate(train_loader):
    # 'data' contains the batch of images
    # Perform operations with the batched data here
    print(f"Batch {batch_idx}: Data shape - {data.shape}")'''

# %%
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

# %%


lr = 1e-3

epochs = 400

beta = .5

input_channels = 1
# %%

from test2 import VAE
model = VAE().to(DEVICE)
#%%
from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mu, logvar):
    #reproduction_loss = nn.functional.binary_cross_entropy(x_hat.float(), x.float(), reduction='sum')
    criterion=nn.MSELoss(reduction='none')
    reconstruction_loss=criterion(x_hat,x)
    #print(reproduction_loss)
    KLD      = - beta * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
    

    return reconstruction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

# %%
print("Start training VAE...")
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, x in enumerate(train_loader):

        #x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mu, logvar = model(x.float())
        loss = loss_function(x.float(), x_hat, mu, logvar)
        #print(loss)
        overall_loss += loss.mean().item()
        
        loss.mean().backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    
print("Finish!!")
# %%
torch.save(model.state_dict(), 'model_weights1.pth')
# %%
print("Start training VAE...")
model.eval()

num_samples = 0
overall_loss = 0
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):

        data = data.to(DEVICE)

        x_hat, mu, logvar = model(data.float())
        total_batch_loss = loss_function(data.float(), x_hat, mu, logvar)

        
        # Accumulate total loss and number of samples
        overall_loss += total_batch_loss.mean().item()
        num_samples += data.size(0)

# Calculate average loss over the entire test set
average_loss = overall_loss / num_samples
print(f"Average Loss on Test Set: {average_loss:.4f}")
# %%
