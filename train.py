#!/usr/bin/env python3

import torch

from book_parser import iterate_paragraph
from loader import create_dataloader

from itertools import islice, batched

from tokenizer import Vocabulary

SEED = 123

voc = Vocabulary('voc.json')

with open('input_dataset/dom_casmurro.txt') as f:
    paragraphs = iterate_paragraph(f)
    raw = ' '.join([line.strip() for line in paragraphs])

context_length = 4
output_dims = 256

dataloader = create_dataloader(
    txt=raw,
    voc=voc,
    max_length=context_length
)

# data_iter = iter(dataloader)
# first = next(data_iter)

torch.manual_seed(SEED)

token_embedding_layer = torch.nn.Embedding(voc.vocab_size(), output_dims)
pos_embeddings_layer = torch.nn.Embedding(context_length, output_dims)

inputs, _ = next(iter(dataloader))
token_embeddings = token_embedding_layer(inputs)
pos_embeddings = pos_embeddings_layer(torch.arange(context_length))
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
