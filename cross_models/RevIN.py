import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        # x: [B, T, D]
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)  # [B, 1, D]
            self.std = x.std(dim=1, keepdim=True) + self.eps  # [B, 1, D]
            x = (x - self.mean) / self.std

            if self.affine:
                x = x * self.gamma + self.beta

            return x

        elif mode == 'denorm':
            x = (x - self.beta) / (self.gamma + self.eps) if self.affine else x
            x = x * self.std + self.mean
            return x

        else:
            raise ValueError("mode must be 'norm' or 'denorm'")
