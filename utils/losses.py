import torch
import torch.nn as nn


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
        loss = torch.mean(-torch.log(d_out_real) - self.gamma * torch.log(torch.ones_like(d_out_fake) - d_out_fake))
        return loss


class PretrainLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, d_out_real, d_out_fake):
        # calculate loss using the function defined in the paper
        loss_real = torch.mean(torch.log2(d_out_real))
        loss_fake = torch.mean(torch.log2(torch.ones_like(d_out_fake) - d_out_fake))
        return -(loss_real + loss_fake) / 2