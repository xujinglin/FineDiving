import torch.nn as nn
from models.PS_parts import *


class PSNet(nn.Module):
    def __init__(self, n_channels=6):
        super(PSNet, self).__init__()

        self.inc = inconv(n_channels, 12)
        self.down1 = down(12, 24)
        self.down2 = down(24, 48)
        self.down3 = down(48, 96)
        self.down4 = down(96, 96)
        self.tas = MLP_tas(64, 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.tas(x5)
        return x5, x