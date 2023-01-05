import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import datetime
now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') 

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
    """
    Loads numpy array stored in .npy format
    :param arr_dir: path to existing binary file
    :return: numpy array
    """
    with open(arr_dir, 'rb') as  f:
        data = np.load(f)
    return data


def save_numpy(arr, arr_dir):
    """
    Saves numpy array as a file in .npy format
    :param arr: numpy array to save
    :param arr_dir: desired path to saved binary file
    :return: none
    """
    with open(arr_dir, 'wb') as f:
        np.save(f, arr)

def save_gan(model, ):
    pass

model = AutoEncoder(protdim=2, locdim=2, aadim=2).cuda()
model.load_state_dict(torch.load('embedding_model_with_zeros.pt'))


def noise_data(n):
    """
    Generate noise data for GAN (FGAN) model
    :param n: number of samples to generate
    :return: random samples from multidimensional normal distribution
    """
    return np.random.normal(0, 8, [n, 256])


def pre_decode(var, prot_dec, aa_dec):
    """
    Decodes variants stored as a matrix back to list of substitutions
    Please note that original amino acid is ignored, only substitution is returned
    :param var: variant as a matrix with encoded substitutions with zero padding to achieve uniform size
    :param prot_dec: inverted protein label encoding dictionary
    :param aa_dec: inverted amino acid label encoding dictionary
    :return:
    """
    var = var[arr[0].sum(axis=1) != 0] # exclude zero padding rows
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
alpha = 0.5
beta = 15
###Discriminator Hyperparameters###
gamma = 0.1

# Loss function
gen_loss = GenFGANLoss(alpha_=alpha, beta_=beta) #generator loss function
disc_loss = DiscFGANLoss(gamma_=gamma) #discriminator loss function

# Initialize generator and discriminator
generator = Generator().cuda()
discriminator = Discriminator().cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=5e-4) #intial learning rates
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

n_epochs = 100
batch_size = 1000
arr_d_loss, arr_g_loss, arr_i = [], [], []
v_animate = 1

hyperparametrs = (alpha, beta, gamma) #alpha-- hyperparameter used to generate data points on deltaX(boundary region), Beta: controlling disperion loss, Gamma: anamoly paramenters used to control discriminator
#When gamma is less than 1, the discriminator will focus more on classifying the real data points correctly.

data = model.encoder(Tensor(arr).reshape(-1, 3, 1)).reshape(-1, 1, 50, 6).detach() #reshaping the array--> for architecture 50(fixed vector value for number of submitutions, eg:8--->50-8=42 zeropadding), 6 is latent domain subsitution
#(prot_dim = loc_dim = aa_dim = 2.)--->2+2+2=6  Protien is represented by vector of dim 2 
n_batches = len(data) // batch_size
training_data = data[:batch_size * n_batches].reshape((n_batches, batch_size, 1, 50, 6)).cuda()
# test_data = training_data[round(n_batches*0.8)+1:]
# training_data = training_data[:round(n_batches*0.8)+1]


for i, epoch in enumerate((range(n_epochs))): #search abt this
    print(f'\nepoch: {epoch+1} / {n_epochs}:\n')
    try:
        g_epoch_loss, d_epoch_loss = 0, 0 
        # Configure input
        for j, real_data_batch in enumerate(tqdm(training_data)):
            fake_data_batch = generator(Tensor(noise_data(batch_size))) #generating fake data

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad() #search abt this

            # Measure discriminator's ability to classify real from generated samples
            d_loss = disc_loss(discriminator(real_data_batch), discriminator(fake_data_batch))

            d_loss.backward() #backprop
            optimizer_D.step() #search abt this 

            # -----------------
            #  Train Generator
            # -----------------
            for _ in range(2):
                fake_data_batch = generator(Tensor(noise_data(batch_size)))
                optimizer_G.zero_grad()

                # Loss measures generator's ability to fool the discriminator
                g_loss = gen_loss(discriminator(fake_data_batch), fake_data_batch)

                g_loss.backward()
                optimizer_G.step()

            arr_g_loss.append(g_loss.item())
            arr_d_loss.append(d_loss.item())


        if i % v_animate == 0: #for every v_animate , the graph will be updated --i.e every iteration 
            #clear() # clear_output(wait=True)
            print(f'generator loss {g_loss.item()}\ndiscriminator loss {d_loss.item()}')
            plt.figure(figsize=(8,6))
            plt.plot(np.arange(len(arr_g_loss))/n_batches, arr_g_loss, label='G')
            plt.plot(np.arange(len(arr_g_loss))/n_batches, arr_d_loss, label='D')
            plt.title(f'Current epoch:{epoch+1}, batch:{j}/{n_batches}\nalpha={alpha},beta={beta},gamma={gamma}')
            plt.ylabel('losses')
            plt.xlabel('epoch')
            plt.plot([0,len(arr_g_loss)/n_batches],[0.693,0.693],color='r',ls='--')
            # plt.plot([0, len(arr_g_loss) / n_batches], [2,2], color='black', ls='--')
            plt.ylim(bottom=0)
            plt.legend()
            plt.show()

    except KeyboardInterrupt:

        print('stop')
        exit()



# torch.save(generator.state_dict(), f'models_saved/fgan_substs_gen-{now}-hyp={hyperparametrs}.pt')
# torch.save(discriminator.state_dict(), f'models_saved/fgan_substs_disc-{now}-hyp={hyperparametrs}.pt')


#if __name__ == '__main__':
