import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, image_size=100, latent_dim=20):
        super(VAE, self).__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim

        # Encoder
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

        # Latent space
        self.fc_mu = nn.Linear(256 * (image_size // 16) * (image_size // 16), latent_dim)
        self.fc_logvar = nn.Linear(256 * (image_size // 16) * (image_size // 16), latent_dim)

        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, 256 * (image_size // 16) * (image_size // 16))

        self.decoder = nn.Sequential(
            #nn.Linear(latent_dim, 256 * (image_size // 16) * (image_size // 16)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, ),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z=self.fc_decoder(z)
        #
        z = z.view(z.size(0), 256, self.image_size // 16, self.image_size // 16)
        #print('here')
        #print(z.size())
        z = self.decoder(z)
        #print(z.size())

        return z.view(-1, 1, self.image_size, self.image_size)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        #print(z.size())
        #z=z.view(-1,256,6,6 )
        x_hat = self.decode(z)
        return x_hat, mu, logvar