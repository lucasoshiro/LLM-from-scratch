#!/usr/bin/env python3

from typing import TextIO
from collections import OrderedDict
from sys import argv, stderr
from bpe import Encoder
from book_parser import iterate_paragraph

class Vocabulary:
    encoder: Encoder
    path: str

    def __init__(self, path):
        self.path = path
        
        try:
            self.deserialize()
        except FileNotFoundError:
            self.encoder = Encoder(pct_bpe=0.88, lowercase=False)

    def deserialize(self) -> None:
        self.encoder = Encoder.load(self.path)

    def extend(self, texts: list[str]) -> None:
        self.encoder.fit(texts)

    def serialize(self) -> None:
        self.encoder.save(self.path)

    def __str__(self) -> str:
        return str(self.encoder)

    def encode(self, s: str) -> list[int | None]:
        return next(self.encoder.transform([s]))

    def decode(self, tokens: list[int | None]) -> str:
        return next(self.encoder.inverse_transform([tokens]))

def populate(voc: Vocabulary, path: str) -> None:
    with open(path) as f:
        paragraphs = iterate_paragraph(f)
        voc.extend([*paragraphs])

if __name__ == '__main__':
    voc = Vocabulary('voc.json')
    _, *args = argv

    for arg in args:
        populate(voc, arg)
        print('populated', arg, file=stderr)

    while True:
        try:
            s = input()
        except:
            break

        encoded = voc.encode(s)
        print(encoded)
        decoded = voc.decode(encoded)
        print(decoded)

    voc.serialize()
