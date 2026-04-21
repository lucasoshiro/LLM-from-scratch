#!/usr/bin/env python3

from typing import TextIO, Generator

def iterate_paragraph(input_f: TextIO) -> Generator[str]:
    last_line = ''

    for line in input_f:
        line = line.strip()

        if line == '':
            yield last_line
            last_line = ''
        else:
            last_line += ' ' + line
    yield last_line

if __name__ == '__main__':
    from sys import argv
    with open(argv[1]) as f:
        for p in iterate_paragraph(f):
            print(p)
