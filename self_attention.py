#!/usr/bin/env python3

import torch
import torch.nn as nn

class SelfAttentention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.wv = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        ks = self.wk(x)
        vs = self.wv(x)
        qs = self.wq(x)

        omegas = qs @ ks.T
        alphas = torch.softmax(omegas / ks.shape[-1] ** 0.5, dim=-1)
        return alphas @ vs

if __name__ == '__main__':
    torch.manual_seed(789)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]
         ]
    )

    sa = SelfAttentention(3, 2)
    f = sa.forward(inputs)
    print(f)
