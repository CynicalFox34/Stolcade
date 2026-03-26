"""
Stolcade neural network — combined policy + value head.
Input:  (batch, 10, 20, 11) board tensor
Output: value scalar in [-1, 1]  (1 = current player wins)
        (Policy head can be added later for MCTS; for now we use value-only TD training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class StokcadeNet(nn.Module):
    def __init__(self, channels=128, res_blocks=12):
        super().__init__()
        # Input tower
        self.input_conv = nn.Conv2d(12, channels, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_tower = nn.Sequential(*[ResBlock(channels) for _ in range(res_blocks)])

        # Policy head (outputs probability distribution over the board)
        self.pol_conv  = nn.Conv2d(channels, 32, 1, bias=False)
        self.pol_bn    = nn.BatchNorm2d(32)
        # We output a scalar for every cell (20x11) representing move "desirability"
        self.pol_linear = nn.Linear(32 * 20 * 11, 20 * 11)

        # Value head (outputs scalar [-1, 1])
        self.val_conv  = nn.Conv2d(channels, 1, 1, bias=False)
        self.val_bn    = nn.BatchNorm2d(1)
        self.val_fc1   = nn.Linear(20 * 11, 64)
        self.val_fc2   = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B, 12, 20, 11)
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_tower(x)

        # Policy head
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = p.flatten(1)
        p = self.pol_linear(p)  # Logits for target squares

        # Value head
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = v.flatten(1)
        v = F.elu(self.val_fc1(v))       # ELU allows negative values (ReLU was clipping them)
        v = torch.tanh(self.val_fc2(v))  # [-1, 1]

        return p, v  # (B, 220), (B,)
