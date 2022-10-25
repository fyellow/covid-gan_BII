import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.AE import AutoEncoder
from utils.variant_GAN import Generator, Discriminator
from utils.losses import *
# from torchsummary import summary

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=UserWarning)

import os

clear = lambda: os.system('clear')  # on Linux System

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def load_numpy(arr_dir):
    with open(arr_dir, 'rb') as f:
        data = np.load(f)
    return data


def save_numpy(arr, arr_dir):
    with open(arr_dir, 'wb') as f:
        np.save(f, arr)


model = AutoEncoder(protdim=2, locdim=2, aadim=2).cuda()
model.load_state_dict(torch.load('embedding_model_with_zeros.pt'))


def noise_data(n):
    return np.random.normal(0, 8, [n, 256])


def pre_decode(var, prot_dec, aa_dec):
    var = var[arr[0].sum(axis=1) != 0]
    res = []
    for row in var:
        res.append(prot_dec[row[0]] + '_' + str(int(row[1])) + aa_dec[row[2]])
    return res


arr_whole = load_numpy('data/variants_1.npy')
metadata = pd.read_csv('data/variants_1_meta.csv')
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

n_epochs = 3 + 1
batch_size = 500
arr_d_loss, arr_g_loss, arr_i = [], [], []
v_animate = 1

hyperparametrs = (alpha, beta, gamma)

data = model.encoder(Tensor(arr).reshape(-1, 3, 1)).reshape(-1, 1, 50, 6).detach()
n_batches = len(data) // batch_size
training_data = data[:batch_size * n_batches].reshape((n_batches, batch_size, 1, 50, 6)).cuda()
# test_data = training_data[round(n_batches*0.8)+1:]
# training_data = training_data[:round(n_batches*0.8)+1]


for i, epoch in enumerate((range(n_epochs))):
    print('epoch', epoch)
    try:
        g_epoch_loss, d_epoch_loss = 0, 0
        # Configure input
        for j, real_data_batch in enumerate(tqdm(training_data)):
            fake_data_batch = generator(Tensor(noise_data(batch_size)).cuda())

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            d_loss = disc_loss(discriminator(real_data_batch), discriminator(fake_data_batch))

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

        if i % v_animate == 0:
            clear() # clear_output(wait=True)
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
        if j == 70:
            break
    except KeyboardInterrupt:
        print('stop')
