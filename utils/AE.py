import torch
import torch.nn as nn

Tensor = torch.FloatTensor


class BasicEncoding(nn.Module):
    def __init__(self, ndim=5):
        super(BasicEncoding, self).__init__()
        self.ndim = ndim
        self.fc1 = nn.Linear(1, self.ndim)
        self.fc2 = nn.Linear(self.ndim, self.ndim)
        self.activation = nn.LeakyReLU()

    def forward(self, Input):
        x = self.activation(self.fc1(Input))
        x = self.activation(self.fc2(x))
        return x


# class BasicDecoding(nn.Module):
#     def __init__(self, indim=5, outdim=1):
#         super(BasicDecoding, self).__init__()
#         self.indim = indim
#         self.outdim = outdim
#         self.fc1 = nn.Linear(self.indim, self.indim)
#         self.fc2 = nn.Linear(self.indim, self.outdim)
#         self.rl = nn.ReLU()
#         self.sm = nn.Softmax(dim=0)
#     def forward(self, Input):
#         if self.outdim == 1:
#             x = self.fc2(self.rl(self.fc1(Input))).round()
#         else:
#             x = torch.argmax(self.sm(self.fc2(self.rl(self.fc1(Input)))), dim=1)
#         return x

class BasicDecoding(nn.Module):
    def __init__(self, indim=5, outdim=1):
        super(BasicDecoding, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.fc1 = nn.Linear(self.indim, self.indim)
        self.fc2 = nn.Linear(self.indim, 1)
        self.rl = nn.LeakyReLU()
        self.sm = nn.Softmax(dim=0)

    def forward(self, Input):
        x = self.fc2(self.rl(self.fc1(Input)))  # .round()
        return x


class SubstitutionEncoding(nn.Module):
    def __init__(self, protdim=5, locdim=5, aadim=5):
        super(SubstitutionEncoding, self).__init__()
        self.protdim = protdim
        self.locdim =locdim
        self.aadim = aadim

        self.prot_encoder = BasicEncoding(ndim=self.protdim)
        self.loc_encoder = BasicEncoding(ndim=self.locdim)
        self.aa_encoder = BasicEncoding(ndim=self.aadim)

    def forward(self, Input):
        prot_encoding = self.prot_encoder(Input[:, 0])
        loc_encoding = self.loc_encoder(Input[:, 1])
        aa_encoding = self.aa_encoder(Input[:, 2])

        x = torch.cat((prot_encoding, loc_encoding, aa_encoding), dim=1)
        return x


class SubstitutionDecoding(nn.Module):
    def __init__(self, protdim=5, locdim=5, aadim=5):
        super(SubstitutionDecoding, self).__init__()
        self.protdim = protdim
        self.locdim = locdim
        self.aadim = aadim

        self.prot_decoder = BasicDecoding(indim=self.protdim, outdim=24)
        self.loc_decoder = BasicDecoding(indim=self.locdim, outdim=1)
        self.aa_decoder = BasicDecoding(indim=self.aadim, outdim=20)

    def forward(self, Input):
        prot_prt = Input[:, :self.protdim]
        loc_prt = Input[:, self.protdim:self.locdim + self.protdim]
        aa_prt = Input[:, self.locdim + self.protdim:]

        n = len(Input)
        prot = self.prot_decoder(prot_prt).reshape(n, 1)
        loc = self.loc_decoder(loc_prt).reshape(n, 1)
        aa = self.aa_decoder(aa_prt).reshape(n, 1)
        x = torch.cat((prot, loc, aa), dim=1).reshape(n, 3, 1)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, protdim=5, locdim=5, aadim=5):
        super(AutoEncoder, self).__init__()
        self.protdim = protdim
        self.locdim = locdim
        self.aadim = aadim

        self.encoder = SubstitutionEncoding(protdim=self.protdim, locdim=self.locdim, aadim=self.aadim)
        self.decoder = SubstitutionDecoding(protdim=self.protdim, locdim=self.locdim, aadim=self.aadim)

    def forward(self, Input):
        x = self.decoder(self.encoder(Input))
        return x
