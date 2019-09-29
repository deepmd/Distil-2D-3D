import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Loss(nn.Module):
    def __init__(self, name, weight, logit_input=True, logit_target=True):
        super(Loss, self).__init__()
        self.input_transform = lambda x: x
        self.target_transform = lambda x: x
        if name == 'L1':
            self.loss = F.l1_loss
        elif name == 'L2':
            self.loss = F.mse_loss
        elif name == 'CE':
            self.loss = F.cross_entropy
            self.input_transform = partial(F.softmax, dim=-1)
            self.target_transform = partial(F.softmax, dim=-1)
        elif name == 'BCE':
            if logit_input and logit_target:
                self.loss = F.binary_cross_entropy_with_logits
            else:
                self.loss = F.binary_cross_entropy
                self.input_transform = F.sigmoid
                self.target_transform = F.sigmoid
        elif name == 'KL_DIV':
            self.loss = partial(F.kl_div, reduction='batchmean')
            # pytorch kl divergence function expects log probabilities as input
            self.input_transform = partial(F.log_softmax, dim=-1)
            self.target_transform = partial(F.softmax, dim=-1)
        else:
            raise ValueError("There's no loss function named '{}'!".format(name))

        if not logit_input:
            self.input_transform = lambda x: x
        if not logit_target:
            self.target_transform = lambda x: x

        self.weight = weight

    def forward(self, input, target):
        input = self.input_transform(input)
        target = self.target_transform(target)
        loss = self.weight * self.loss(input, target)
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
