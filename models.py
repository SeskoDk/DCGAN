from typing import Tuple

import torch
from torch import nn
from utils import weight_init, Flatten


class Generator(nn.Module):
    def __init__(self, nz: int = 100, nc: int = 3, ngf: int = 64) -> None:
        """
        :param nz: Size of z latent vector
        :param nc: Number of channels in the training images
        :param ngf: Size of feature maps in generator
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input: B x (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: B x (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: B x (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: B x (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: B x (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Shape: B x (nc) x 64 x 64
        )
        self.apply(weight_init)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc: int = 3, ndf: int = 64):
        """
        :param nc: # Number of channels in the training images.
        :param ndf: Size of feature maps in discriminator
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input: B x (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: B x (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: B x (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: B x (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: B x (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # Shape: B x 1
            Flatten()
        )
        self.apply(weight_init)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.main(input)


def main():
    batch = torch.rand(4, 100, 1, 1)
    generator = Generator()
    discriminator = Discriminator()

    gen_output = generator(batch)
    print(gen_output.shape)

    dis_output = discriminator(gen_output)
    print(dis_output.shape)

    print(generator)
    print(discriminator)


if __name__ == "__main__":
    main()
