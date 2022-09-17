import torch
from torch import nn


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.BCE = torch.nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs, targets):
        inputs = inputs[:, 1, :, :]
        inputs = self.sigmoid(inputs)
        targets = targets.float()
        bce = self.BCE(inputs, targets)
        return bce
