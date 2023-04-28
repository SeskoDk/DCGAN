import torch.cuda
from torch import nn


class Generator(nn.Module):
    def __init__(self, nz: int = 100, nc: int = 3, ngf: int = 64) -> None:
        """
        :param nz: Size of z latent vector
        :param nc: Number of channels in the training images
        :param ngf: Size of feature maps in generator
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)

m = nn.ConvTranspose2d()
batch = torch.rand(size=(4, 3, 64, 64))
