import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 256

        self.fc_1 = nn.Linear(self.latent_dim, self.latent_dim * 2 * 2)
        self.bn_fc = nn.BatchNorm2d(self.latent_dim)

        self.convt_1 = nn.ConvTranspose2d(self.latent_dim, 128, (6, 2), stride=(1, 1))
        self.bn_1 = nn.BatchNorm2d(128)

        self.convt_2 = nn.ConvTranspose2d(128, 32, (6, 2), stride=(2, 1))
        self.bn_2 = nn.BatchNorm2d(32)

        self.convt_3 = nn.ConvTranspose2d(32, 16, (6, 2), stride=(1, 1))
        self.bn_3 = nn.BatchNorm2d(16)

        self.convt_4 = nn.ConvTranspose2d(16, 8, (3, 2), stride=(2, 1))
        self.bn_4 = nn.BatchNorm2d(8)

        self.convt_5 = nn.ConvTranspose2d(8, 1, (4, 1), stride=(1, 1))
        self.LRe = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, Input):
        x = self.fc_1(Input).reshape(-1, self.latent_dim, 2, 2)
        x = self.LRe(self.bn_fc(x))

        x = self.convt_1(x)
        x = self.LRe(self.bn_1(x))

        x = self.convt_2(x)
        x = self.LRe(self.bn_2(x))

        x = self.convt_3(x)
        x = self.LRe(self.bn_3(x))

        x = self.convt_4(x)
        x = self.LRe(self.bn_4(x))

        x = self.convt_5(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc_1 = nn.Linear(64 * 2 * 2, 1)
        self.activation = nn.Sigmoid()

        self.convt_1 = nn.Conv2d(1, 16, (7, 2), stride=(1, 1))
        self.bn_1 = nn.BatchNorm2d(16)

        self.convt_2 = nn.Conv2d(16, 32, (6, 2), stride=(2, 1))
        self.bn_2 = nn.BatchNorm2d(32)

        self.convt_3 = nn.Conv2d(32, 64, (6, 2), stride=(2, 1))
        self.bn_3 = nn.BatchNorm2d(64)

        self.convt_4 = nn.Conv2d(64, 64, (6, 2), stride=(2, 1))
        self.bn_4 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.convt_1(x)
        x = self.bn_1(x)

        x = self.convt_2(x)
        x = self.bn_2(x)

        x = self.convt_3(x)
        x = self.bn_3(x)

        x = self.convt_4(x)
        x = self.bn_4(x)

        x = self.activation(self.fc_1(torch.flatten(x, 1)))
        return (x)
