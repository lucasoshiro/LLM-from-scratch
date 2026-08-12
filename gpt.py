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

    def forward(self, x):
        return x

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
            torch.arrange(seq_len, device=x.device)
        )
        embs = self.drop_emb(tok_embs + pos_embs)
        transf = self.transf_blocks(embs)
        norm = self.final_norm(transf)
        logits = self.out(norm)
        return logits
        


if __name__ == '__main__':
    cfg = {
        'vocab_size': 50257,
        'context_length': 1024,
        'emb_dim': 768,
        'n_heads': 12,
        'n_layers': 12,
        'drop_rate': 0.1,
    }

    gpt = GPTModel(cfg)
