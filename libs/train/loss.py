import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Loss(nn.Module):
    def __init__(self, name, weight):
        super(Loss, self).__init__()
        if name == 'L1':
            self.loss = F.l1_loss
        elif name == 'L2':
            self.loss = F.mse_loss
        elif name == 'CE':
            self.loss = F.cross_entropy
        elif name == 'BCE':
            self.loss = F.binary_cross_entropy_with_logits
        else:
            raise ValueError("There's no loss function named '{}'!".format(name))
        self.weight = weight

    def forward(self, inputs, targets):
        loss = self.weight * self.loss(inputs, targets)
        return loss


class Regularizer(nn.Module):
    def __init__(self, name, weight):
        super(Regularizer, self).__init__()
        if name == 'L1':
            self.reg = partial(torch.norm, p=1)
        elif name == 'L2':
            self.reg = partial(torch.norm, p=2)
        else:
            raise ValueError("There's no regularizer named '{}'!".format(name))
        self.weight = weight

    def forward(self, parameters):
        loss = self.weight * sum(self.reg(param) for param in parameters)
        return loss
