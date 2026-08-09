#!/usr/bin/env python3

import torch
import torch.nn as nn

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

        if d_out % num_heads != 0:
            raise ValueError('d_out must be multiple of num_heads')

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.wqs = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wks = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wvs = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), 1)
        )

    def forward(self, x):
        b, n, d_in = x.shape
            
        ks = self.wks(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        vs = self.wvs(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        qs = self.wqs(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        omegas = qs @ ks.transpose(2, 3)
        masked = omegas.masked_fill(self.mask.bool()[:n, :n], -torch.inf)
        alphas = torch.softmax(masked / ks.shape[-1] ** 0.5, dim=-1)
        alphas = self.dropout(alphas)

        zs = (alphas @ vs).transpose(1, 2).contiguous().view(b, n, self.d_out)
        return self.out_proj(zs)

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
