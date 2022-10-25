import torch
loss = torch.nn.BCELoss()(torch.tensor([1 / 2]), torch.tensor([1.]))
print(loss.item())
