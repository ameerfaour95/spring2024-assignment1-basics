from typing import Iterable
import numpy as np
import time

from cs336_basics.tokenizer import Tokenizer

tinystory = {
    'train':'data/raw/TinyStoriesV2-GPT4-train.txt',
    'val':'data/raw/TinyStoriesV2-GPT4-valid.txt',
    'vocab_filepath': 'data/out/tinystories_vocab.json',
    'merges_filepath': 'data/out/tinystories_merges.txt',
    'special_tokens': ['<|endoftext|>']
}


tokenizer = Tokenizer.from_files(**tinystory)

for split in ['train', 'val']:
    with open(tinystory[split]) as f:
        text = f.read()
    encoded = tokenizer.encode(text)
    arr = np.memmap(f'data/ts/{split}.bin', dtype=np.uint16, mode='w+', shape=(len(encoded),))
    arr[:] = np.array(encoded)
arr.flush()