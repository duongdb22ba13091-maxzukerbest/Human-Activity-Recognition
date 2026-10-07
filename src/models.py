
import torch
import torch.nn as nn


class MLP_HAR(nn.Module):
    """
    Multi-Layer Perceptron for Human Activity Recognition.

    Architecture:
        561 -> 256 -> 128 -> 6
    """

    def __init__(self):
        super(MLP_HAR, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(561, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.network(x)
