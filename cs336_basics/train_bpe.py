# pytest /home/groups/candes/zitong/cs336-assignment1-basics/tests/test_train_bpe.py
import regex as re
from typing import Iterable

from cs336_basics.utils import GPT2_PRETOKENIZER_PATTERN

def _contain_pair(byte_tuple: Iterable[bytes], byte_pair: Iterable[bytes]):
    """
    Check if the byte pair is in the byte tuple.
    If so, return a dictionary with the prefix, pair, and suffix.
    """
    for i in range(len(byte_tuple) - 1):
        if byte_tuple[i:i+2] == byte_pair:
            return dict(prefix=byte_tuple[:i], middle=byte_pair, suffix=byte_tuple[i+2:])
    return None
                

def train_bpe(input_path: str, vocab_size: int, special_tokens: Iterable[str]):
    """
    Train a byte pair encoding tokenizer on the input text file.

    Args:
        input_path: Path to the input text file.
        vocab_size: Size of the vocabulary.
        special_tokens: List of special tokens to add to the vocabulary.

    Returns:
        Tuple of the learned vocab and merges.
    """
    # Read the input text file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Initialize the vocab with 256 bytes and sepcial tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[265+i] = token.encode("utf-8")

    # Remove special tokens from teh text
    for token in special_tokens:
        text = text.replace(token, "")
    
    # Initialize pre-token frequency table
    pretoken_freq = {}
    for pretoken in re.findall(GPT2_PRETOKENIZER_PATTERN, text):
        pretoken_tuple = tuple([bytes([b]) for b in tuple(pretoken.encode("utf-8"))])
        if pretoken_tuple not in pretoken_freq:
            pretoken_freq[pretoken_tuple] = 0
        pretoken_freq[pretoken_tuple] += 1
    
    # Initialize pair frequency table
    pair_freq = {}
    for pretoken_tuple, freq in pretoken_freq.items():
        for i in range(len(pretoken_tuple) - 1):
            pair = pretoken_tuple[i:i+2]
            pair_freq[pair] = pair_freq.get(pair, 0) + freq
        
    # Initialize the merges list
    merges = []

    # Perform the BPE algorithm
    while len(vocab) < vocab_size:
        # Find the most frequent pair
        most_freq_pair = max(pair_freq, key=pair_freq.get)

        # Add the pair to the merges list
        merges.append(most_freq_pair)
        
        # Update the vocab
        new_id = max(vocab.keys()) + 1
        vocab[new_id] = b"".join(most_freq_pair)

        # Update the pre-token frequency table and pair frequency table
        new_pretoken_freq = {}
        for pretoken_tuple, freq in pretoken_freq.items():
            overlap = _contain_pair(pretoken_tuple, most_freq_pair)
            if not overlap:
                new_pretoken_freq[pretoken_tuple] = freq
            else:
                # Update the pre-token frequency table
                new_pretoken = overlap["prefix"] + (vocab[new_id],) + overlap["suffix"]
                new_pretoken_freq[new_pretoken] = freq

                # Update the pair frequency table
                if overlap["prefix"]:
                    add_pair = (overlap["prefix"][-1], vocab[new_id])
                    pair_freq[add_pair] = freq
                    del_pair = (overlap["prefix"][-1], most_freq_pair[0])
                    pair_freq[del_pair] -= freq
                if overlap["suffix"]:
                    add_pair = (vocab[new_id], overlap["suffix"][0])
                    pair_freq[add_pair] = freq
                    del_pair = (most_freq_pair[1], overlap["suffix"][0])
                    pair_freq[del_pair] -= freq
                pair_freq[most_freq_pair] -= freq
        print(pair_freq[most_freq_pair])
        pretoken_freq = new_pretoken_freq
    
    return vocab, merges
        

    

    
if __name__ == '__main__':
    # print(_contain_pair((b'ads', b'q', b'bs', b'c', b'd'), (b'bs', b'c'))) 
    # txt_path = '/home/groups/candes/zitong/cs336-assignment1-basics/tests/fixtures/tinystories_sample_5M.txt'
    # special_tokens = ['<|endoftext|>']
    # vocab_size = 100*(10**3)
    # train_bpe(txt_path, vocab_size, special_tokens)

    txt_path = '/home/groups/candes/zitong/cs336-assignment1-basics/cs336_basics/test_text.txt'
    special_tokens = ['<|endoftext|>']
    vocab_size = 260
    train_bpe(txt_path, vocab_size, special_tokens)