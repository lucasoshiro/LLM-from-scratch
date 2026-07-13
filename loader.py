#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import Vocabulary

class LLMDataset(Dataset):
    input_ids: list[torch.Tensor]
    target_ids: list[torch.Tensor]

    def __init__(
            self,
            txt: str,
            voc: Vocabulary,
            max_length: int,
            stride: int
    ):

        self.input_ids = []
        self.target_ids = []

        token_ids = voc.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            output_chunk = token_ids[i+1 : i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(
        txt: str,
        voc: Vocabulary,
        batch_size: str = 4,
        max_length: int = 256,
        stride: int = 128,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
):
    dataset = LLMDataset(txt, voc, max_length, 1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

