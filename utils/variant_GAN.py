import torch
import torch.nn as nn


class Generator(nn.Module): #noise i/p image o/p
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = 256 #length of generator's latent space--abritary? 

        self.fc_1 = nn.Linear(self.latent_dim, self.latent_dim * 2 * 2) # takes input noise vector-self.latnet_dim
        #output tensor has enough dimensions to be reshaped into a 2D feature map that can be processed by the convolutional layers
        self.bn_fc = nn.BatchNorm2d(self.latent_dim) 

        self.convt_1 = nn.ConvTranspose2d(self.latent_dim, 128, (6, 2), stride=(1, 1))
        self.bn_1 = nn.BatchNorm2d(128) #improves training process- covariant shift---maynot be necessary cause our i/p is randnoise that is normalizised. 

        self.convt_2 = nn.ConvTranspose2d(128, 32, (6, 2), stride=(2, 1))
        #i/p-128, o/p:32 kernel size: (6,2) 
        self.bn_2 = nn.BatchNorm2d(32)

        self.convt_3 = nn.ConvTranspose2d(32, 16, (6, 2), stride=(1, 1))
        self.bn_3 = nn.BatchNorm2d(16)
        #number of filters changes-help to reduce the number of parameters in the network and prevent overfitting.

        self.convt_4 = nn.ConvTranspose2d(16, 8, (3, 2), stride=(2, 1))#smaller strides: larger output feature map-higher spartail reso
        self.bn_4 = nn.BatchNorm2d(8)

        self.convt_5 = nn.ConvTranspose2d(8, 1, (4, 1), stride=(1, 1))
        self.LRe = nn.LeakyReLU(negative_slope=0.2) #(0.01 to 0.3)

    def forward(self, Input):
        x = self.fc_1(Input).reshape(-1, self.latent_dim, 2, 2) #-1: divides dimensions of np array into 2x2 matrices  self.latentdim 
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
        self.activation = nn.Sigmoid() #binary clasification 

        self.convt_1 = nn.Conv2d(1, 16, (7, 2), stride=(1, 1)) #filter size-(7,2)
        self.bn_1 = nn.BatchNorm2d(16)

        self.convt_2 = nn.Conv2d(16, 32, (6, 2), stride=(2, 1))
        self.bn_2 = nn.BatchNorm2d(32)

        self.convt_3 = nn.Conv2d(32, 64, (6, 2), stride=(2, 1))
        self.bn_3 = nn.BatchNorm2d(64)

        self.convt_4 = nn.Conv2d(64, 64, (6, 2), stride=(2, 1))
        self.bn_4 = nn.BatchNorm2d(64) #why 64? 

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
#with dropoutlayers
    # class Generator(nn.Module):
    #def __init__(self, latent_dim=256):
     #   super(Generator, self).__init__()
      #  self.latent_dim = latent_dim
#
 #       self.fc1 = nn.Linear(latent_dim, 1024)
  #      self.fc2 = nn.Linear(1024, 50*6)

   #     self.dropout = nn.Dropout(p=0.5)  # Add dropout layer

    #def forward(self, z):
     #   x = F.leaky_relu(self.fc1(z))
      #  x = self.dropout(x)  # Add dropout layer
       # x = self.fc2(x)
        #return x.view(-1, 1, 50, 6)

"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)

        self.dropout = nn.Dropout(p=0.5)  # Add dropout layer

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = self.dropout(x)  # Add dropout layer
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.dropout(x)  # Add dropout layer
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        x = self.dropout(x)  # Add dropout layer
        x = F.leaky_relu(self.conv4(x), negative_slope=0.1)
        x = self.dropout(x)  # Add dropout layer
        x = F.leaky_relu(self.conv5(x), negative_slope=0.1)
        x = self.dropout(x)  # Add dropout layer
        x = torch.sigmoid(self.conv6(x))

        return x.view(-1, 1)
"""
