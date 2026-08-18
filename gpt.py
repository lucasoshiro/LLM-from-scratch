#!/usr/bin/env python3

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x

class LayerNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.transf_blocks = nn.Sequential(
            *(
                TransformerBlock(cfg)
                for _ in range(cfg['n_layers'])
            )
        )

        self.final_norm = LayerNormalization(cfg['emb_dim'])
        self.out = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        

    def forward(self, x):
        batch_size, n = x.shape
        tok_embs = self.tok_emb(x)
        pos_embs = self.pos_emb(
            torch.arange(n, device=x.device)
        )
        embs = self.drop_emb(tok_embs + pos_embs)
        transf = self.transf_blocks(embs)
        norm = self.final_norm(transf)
        logits = self.out(norm)
        return logits
        


if __name__ == '__main__':
    import tiktoken

    tokenizer = tiktoken.get_encoding('gpt2')

    cfg = {
        'vocab_size': 50257,
        'context_length': 1024,
        'emb_dim': 768,
        'n_heads': 12,
        'n_layers': 12,
        'drop_rate': 0.1,
    }

    torch.manual_seed(123)

    gpt = GPTModel(cfg)

    txt1 = 'Every effort moves you'
    txt2 = 'Every day holds a'

    batch = torch.stack([
        torch.tensor(tokenizer.encode(txt))
        for txt in [txt1, txt2]
    ])

    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    ln = LayerNormalization(emb_dim=5)
    out_ln = ln(batch_example)

    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True, unbiased=False) 
    torch.set_printoptions(sci_mode=False)
    print(mean, var)

    # print(gpt(batch).shape)
