#!/usr/bin/env python3

from typing import TextIO
from collections import OrderedDict
import re

def split(s: str) -> list[str]:
    SPLIT_REGEX = r'[,.?!();:"\']|\s|--'
    return [
        token
        for token in re.split(SPLIT_REGEX, s)
        if token
    ]

class Vocabulary:
    path: str
    vocab: dict[str, int]
    vocab_list: list[str]

    def __init__(self, path):
        self.path = path
        self.vocab = {}
        self.vocab_list = []

        try:
            self.deserialize()
        except FileNotFoundError:
            self.extend(['<|endoftext|>'])

    def deserialize(self) -> None:
        with open(self.path) as f:
            self.extend([
                line.strip()
                for line in f
            ])

    def extend(self, tokens: list[str]) -> None:
        vocab_size = len(self.vocab)
        new_tokens = {
            token: vocab_size + i
            for i, token in enumerate({*tokens} - {*self.vocab})
        }
        self.vocab |= new_tokens
        self.vocab_list = [*self.vocab]

    def serialize(self) -> None:
        with open(self.path, 'w') as f:
            for token in self.vocab:
                print(token, file=f)

    def __str__(self) -> str:
        return str(self.vocab)

if __name__ == '__main__':
    voc = Vocabulary('voc.txt')
    print(voc)

    while True:
        try:
            s = input()
        except:
            break

        splitted = split(s)
        voc.extend(splitted)
        print(voc)

    voc.serialize()
