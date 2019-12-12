import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class Loss(nn.Module):
    """ Provide different types of loss functions.
    Args:
        name (string): the name of loss function: 'L1' | 'L2' | 'CE' | 'BCE' | 'KL_DIV'
        weight (float): the weight of this loss in total loss
        logit_input (bool): if True, the input values are considered as logits otherwise as probabilities. (default: True)
            The value of this argument is ignored for L1 and L2 losses.
        logit_target (bool): if True, the target values are considered as logits otherwise as probabilities. (default: True)
            The value of this argument is ignored for L1 and L2 losses.
    """
    def __init__(self, name, weight, logit_input=True, logit_target=True):
        super(Loss, self).__init__()
        self.input_transform = lambda x: x
        self.target_transform = lambda x: x
        if name == 'L1':
            self.loss = F.l1_loss
        elif name == 'L2':
            self.loss = F.mse_loss
        elif name == 'CE':
            if logit_input:
                self.loss = F.cross_entropy
            else:
                self.loss = F.nll_loss
                # Pytorch negative log likelihood function expects log-probabilities as input
                self.input_transform = torch.log
            # Pytorch cross entropy and negative log likelihood functions expect class indices as target
            if logit_target:
                self.target_transform = lambda x: torch.argmax(F.softmax(x, dim=-1), dim=-1)
            else:
                self.target_transform = partial(torch.argmax, dim=-1)
        elif name == 'BCE':
            if logit_input:
                self.loss = F.binary_cross_entropy_with_logits
            else:
                self.loss = F.binary_cross_entropy
            if logit_target:
                self.target_transform = F.sigmoid
        elif name == 'KL_DIV':
            self.loss = partial(F.kl_div, reduction='batchmean')
            # Pytorch KL divergence function expects log-probabilities as input and probabilities as target
            if logit_input:
                self.input_transform = partial(F.log_softmax, dim=-1)
            else:
                self.input_transform = torch.log
            if logit_target:
                self.target_transform = partial(F.softmax, dim=-1)
        else:
            raise ValueError("There's no loss function named '{}'!".format(name))

        self.weight = weight

    def forward(self, input, target):
        input = self.input_transform(input)
        target = self.target_transform(target)
        loss = self.weight * self.loss(input, target)
        return loss


class Regularizer(nn.Module):
    """ Provide different types of parameter regularization functions.
        Args:
            name (string): the name of regularization function: 'L1' | 'L2'
            weight (float): the weight of this regularization in total regularization
        """
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
