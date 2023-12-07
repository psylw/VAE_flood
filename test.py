import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST  # Importing MNIST for example
from torchvision import transforms
from tqdm import tqdm

# Define the VAE architecture for 100x100 images
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Mean and log variance layers
        self.fc_mu = nn.Linear(256 * 6 * 6, 300)
        self.fc_logvar = nn.Linear(256 * 6 * 6, 300)

        # Decoder layers
        self.decoder_input = nn.Linear(300, 256 * 6 * 6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = x.view(-1, 256 * 6 * 6)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        # Decoder
        z = self.decoder_input(z)
        z = z.view(-1, 256, 6, 6)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

# Define transformations and load data
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.round(x)),  # Convert to binary (0 or 1)
])

# Download and load MNIST dataset as an example
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Create the VAE model
vae = VAE()

# Define the loss function (using Binary Cross Entropy loss)
def loss_function(reconstructed_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Define optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
epochs = 2
# Training loop
vae.train()
for epoch in range(epochs):  # Replace 'epochs' with the number of epochs
    total_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        reconstructed, mu, logvar = vae(data)
        loss = loss_function(reconstructed, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader.dataset):.4f}")
