import torch
import torch.nn as nn
import numpy as np


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(256 * 7 * 7, 300)  # Mean
        self.fc_logvar = nn.Linear(256 * 7 * 7, 300)  # Log Variance

        # Decoder layers
        self.decoder_input = nn.Linear(300, 256 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=0),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder(x)
        print(np.shape(x))
        x = x.view(-1, 256 * 7 * 7)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 256, 7, 7)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        print(np.size(mu))
        print(logvar.size())
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        print(x_hat.size())
        return x_hat.squeeze(dim=1), mu, logvar





