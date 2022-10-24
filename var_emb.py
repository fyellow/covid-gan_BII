import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(context="talk")

from tqdm import tqdm
from IPython.display import clear_output

from AE import AutoEncoder
#from torchsummary import summary
import re

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def load_numpy(arr_dir):
    with open(arr_dir, 'rb') as f:
        data = np.load(f)
    return data

def save_numpy(arr, arr_dir):
    with open(arr_dir, 'wb') as f:
        np.save(f, arr)


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


model = AutoEncoder(protdim=2, locdim=2, aadim=2)
model.load_state_dict(torch.load('embedding_model_with_zeros.pt'))


class GenFGANLoss(nn.Module):
    def __init__(self, alpha_=1, beta_=0, **kwargs):
        super().__init__()
        self.alpha = alpha_
        self.beta = beta_

    def forward(self, d_out, g_out):
        # calculate loss using the function defined in the paper
        bce = torch.nn.BCELoss()
        EL = bce(d_out, self.alpha * torch.ones_like(d_out))
        mu = torch.ones_like(g_out) * g_out.mean(dim=0)
        DL = 1 / torch.mean(torch.norm(g_out - mu, p=2))
        loss = EL + self.beta * DL
        return loss


class DiscFGANLoss(nn.Module):
    def __init__(self, gamma_=1, **kwargs):
        super().__init__()
        self.gamma = gamma_

    def forward(self, d_out_real, d_out_fake):
        # calculate loss using the function defined in the paper
        loss = torch.mean(-torch.log2(d_out_real) - self.gamma * torch.log2(torch.ones_like(d_out_fake) - d_out_fake))
        return loss


class PretrainLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, d_out_real, d_out_fake):
        # calculate loss using the function defined in the paper
        loss_real = torch.mean(torch.log2(d_out_real))
        loss_fake = torch.mean(torch.log2(torch.ones_like(d_out_fake) - d_out_fake))
        return -(loss_real + loss_fake) / 2


def noise_data(n):
    return np.random.normal(0, 8, [n, 256])


def pre_decode(var, prot_dec, aa_dec):
    var = var[arr[0].sum(axis=1)!=0]
    res = []
    for row in var:
        res.append(prot_dec[row[0]]+'_'+str(int(row[1]))+aa_dec[row[2]])
    return res

arr_whole = load_numpy('variants_1.npy')
metadata = pd.read_csv('variants_1_meta.csv')
idx = int(1e5)
arr = arr_whole[:idx]
metadata = metadata[:idx]

###Generator Hyperparameters###
alpha = 1
beta = 0
###Discriminator Hyperparameters###
gamma = 1

# Loss function
gen_loss = GenFGANLoss(alpha_=alpha, beta_=beta)
disc_loss = DiscFGANLoss(gamma_=gamma)

# Initialize generator and discriminator
generator = Generator().cuda()
discriminator = Discriminator().cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-3)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

n_epochs=0 +1
batch_size= 500
arr_d_loss, arr_g_loss, arr_i = [], [], []
v_animate=1
# pictures_path = './pictures' + f'_al={alpha}_bt={beta}_gm={gamma}'
# if os.path.exists(pictures_path):
#     rmtree(pictures_path)
# os.makedirs(pictures_path)
hyperparametrs = (alpha, beta, gamma)

data = model.encoder(Tensor(arr).reshape(-1,3,1)).reshape(-1,1,50,6).detach()
n_batches = len(data)//batch_size
training_data = data[:batch_size*n_batches].reshape((n_batches, batch_size, 1, 50, 6)).cuda()
# test_data = training_data[round(n_batches*0.8)+1:]
# training_data = training_data[:round(n_batches*0.8)+1]


for i, epoch in enumerate((range(n_epochs))):
    try:
        g_epoch_loss, d_epoch_loss=0,0
        # Configure input
        for j, real_data_batch in enumerate(tqdm(training_data)):
            fake_data_batch = generator(Tensor(noise_data(batch_size)).cuda())

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            d_loss = disc_loss(discriminator(real_data_batch),discriminator(fake_data_batch))

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            for _ in range(3):
                fake_data_batch = generator(Tensor(noise_data(batch_size)).cuda())
                optimizer_G.zero_grad()

                # Loss measures generator's ability to fool the discriminator
                g_loss = gen_loss(discriminator(fake_data_batch), fake_data_batch)

                g_loss.backward()
                optimizer_G.step()

            arr_g_loss.append(g_loss.item())
            arr_d_loss.append(d_loss.item())

            if j % v_animate == 0:
                clear_output(wait=True)
                print(f'epoch {epoch} \ngenerator loss {g_loss.item()}\ndiscriminator loss {d_loss.item()}')
                plt.figure(figsize=(8,6))
                plt.plot(np.arange(j+1)*(i+1)/n_batches, arr_g_loss, label='G')
                plt.plot(np.arange(j+1)*(i+1)/n_batches, arr_d_loss, label='D')
                plt.title(f'Current epoch:{epoch+1}, batch:{j}/{n_batches}')
                plt.ylabel('losses')
                plt.xlabel('epoch')
                plt.ylim(bottom=0)
                plt.legend()
                plt.show()
            # if j == 70:
            #     break
    except KeyboardInterrupt:
        pass


