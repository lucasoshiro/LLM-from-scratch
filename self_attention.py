#!/usr/bin/env python3

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.5, qkv_bias=False):
        super().__init__()

        self.d_out = d_out

        self.wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, d_in = x.shape

        ks = self.wk(x)
        vs = self.wv(x)
        qs = self.wq(x)

        omegas = qs @ ks.transpose(1, 2)
        # mask = torch.triu(torch.ones(omegas.shape), 1).bool()
        masked = omegas.masked_fill(self.mask.bool()[:n, :n], -torch.inf)
        alphas = torch.softmax(masked / ks.shape[-1] ** 0.5, dim=-1)
        alphas = self.dropout(alphas)
        return alphas @ vs

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            context_length,
            dropout=0.5,
            qkv_bias=False,
            num_heads=2
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

if __name__ == '__main__':
    torch.manual_seed(123)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]
         ]
    )

    batch = torch.stack((inputs, inputs), dim=0)

    sa = MultiHeadAttention(3, 2, batch.shape[1], dropout=0.0)
    f = sa(batch)
    print(f)
