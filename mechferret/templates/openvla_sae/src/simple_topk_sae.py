"""Minimal Top-K SAE implementation for OpenVLA activation experiments.

This is deliberately dependency-light. For serious runs, compare against SAELens/Prisma implementations.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.out_bias = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.encoder(x - self.pre_bias))
        if self.k < z.shape[-1]:
            vals, idx = torch.topk(z, self.k, dim=-1)
            sparse = torch.zeros_like(z)
            sparse.scatter_(-1, idx, vals)
            z = sparse
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.out_bias

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        aux = {
            "l0": (z > 0).float().sum(dim=-1).mean(),
            "mse": F.mse_loss(x_hat, x),
        }
        return x_hat, z, aux
