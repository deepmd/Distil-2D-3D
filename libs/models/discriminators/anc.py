import torch.nn as nn


class ANCDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(ANCDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        validity = self.model(input)
        return validity