"""Implementation of a VAE for the MNIST dataset.

Examples:
    $ python3 vae.py --mse=1
    $ python3 vae.py --mse=0 --sigmoid=1
"""
from dataclasses import dataclass

from datacli import datacli
import torchvision
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TrainConfig:
    mse: int = 1  # True: MSE loss, False: cross-entropy loss
    data_path: str = '/home/iron-man/.cache/torch/mnist/'
    latent_dim: int = 64  # latent variable dimension
    sigmoid: int = 0  # True: sigmoid output, False: linear output
    lr: float = 0.001  # learning rate
    batch_size: int = 16
    epochs: int = 10
    print_steps: int = 500
    device: str = 'cuda'

    def __post_init__(self):
        self.mse = bool(self.mse)
        self.sigmoid = bool(self.sigmoid)
        if self.mse and self.sigmoid:
            raise ValueError('MSE loss and sigmoid output are incompatible.')


class VAE(nn.Module):

    def __init__(self, latent_dim=64, sigmoid=False):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.sigmoid = sigmoid

        # encoder
        # 2 convolution layers
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(2, 2),
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=(2, 2),
        )
        # fully connected to mean, and log-variance (break apart activations later)
        flatten_dims = 64 * 6**2
        self.ff = nn.Linear(flatten_dims, latent_dim * 2)

        # decoder
        # up-sample
        self.upsize = nn.Linear(latent_dim, 32 * 7**2)  # reshape after
        self.unconv1 = nn.ConvTranspose2d(in_channels=32,
                                          out_channels=64,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1)
        self.unconv2 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=32,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1)
        self.unconv3 = nn.ConvTranspose2d(in_channels=32,
                                          out_channels=1,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          output_padding=0)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        # flatten
        h = h.view((h.shape[0], -1))

        h = self.ff(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(x.device)
        # reparameterization trick
        z = mu + std * eps
        z = F.relu(self.upsize(z))
        # reshape
        z = z.view((z.shape[0], 32, 7, 7))
        z = F.relu(self.unconv1(z))
        z = F.relu(self.unconv2(z))
        z = self.unconv3(z)
        if self.sigmoid:
            z = torch.sigmoid(z)
        return z, mu, logvar


def loss_function(y, x, mu, logvar, mse=True):
    if mse:
        ERR = 0.5 * F.mse_loss(y, x, reduction='sum')
    else:
        ERR = F.binary_cross_entropy(y, x, reduction='sum')
    KLD = 0.5 * torch.sum(-logvar - 1 + mu.pow(2) + logvar.exp())
    return ERR + KLD, ERR, KLD


def main(args):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(args.data_path,
                                            transform=transform,
                                            download=True)
    test_data = torchvision.datasets.MNIST(args.data_path,
                                           train=False,
                                           transform=transform,
                                           download=True)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    device = args.device if torch.cuda.is_available() else 'cpu'
    model = VAE(args.latent_dim, args.sigmoid)
    model.to(device)
    # optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for idx, (x, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            x = x.to(device)
            y, mu, logvar = model(x)
            loss = loss_function(y, x, mu, logvar, mse=args.mse)
            loss[0].backward()
            optimizer.step()

            if idx % args.print_steps == 0:
                print(f"Epoch {epoch}, step {idx:,}, total loss: {loss[0].item():.2f}, "
                      f"reconstruction loss: {loss[1].item():.2f}, KLD: {loss[2].item():.2f}")
                # save image alongside generated image
                torchvision.utils.save_image(x, 'input.png')
                torchvision.utils.save_image(y, 'output.png')

    # save image alongside generated image
    x, labels = next(iter(test_loader))
    y, mu, std = model(x.to(device))
    torchvision.utils.save_image(x, 'input.png')
    torchvision.utils.save_image(y, 'output.png')


if __name__ == '__main__':
    args = datacli(TrainConfig)
    print(args)
    main(args)
