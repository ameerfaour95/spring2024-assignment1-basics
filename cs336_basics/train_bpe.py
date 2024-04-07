import regex as re
from typing import Iterable

from utils import GPT2_PRETOKENIZER_PATTERN


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
    
    # Perform pre-tokenization and output the frequency table of the tokens
    pretoken_freq = {}
    for pretoken in re.findall(GPT2_PRETOKENIZER_PATTERN, text):
        pretoken_encoded = pretoken.encode("utf-8")
        if pretoken_encoded not in pretoken_freq:
            pretoken_freq[pretoken_encoded] = 0
        pretoken_freq[pretoken_encoded] += 1
        
    # Initialize the merges list
    merges = []

    # Perform the BPE algorithm
    for i in range(vocab_size):
        # Find the most frequent pair of tokens
        max_freq = 0
        max_pair = None
        for pair in pretoken_freq:
            if pretoken_freq[pair] > max_freq:
                max_freq = pretoken_freq[pair]
                max_pair = pair
        
        # Add the pair to the merges list
        merges.append(max_pair)
        
        # Update the frequency table
        new_pair = max_pair[0] + max_pair[1]
        pretoken_freq[new_pair] = 0
        for pretoken in pretoken_freq:
            pretoken_freq[pretoken] = pretoken_freq[pretoken].replace(max_pair, new_pair)
    


    

    
if __name__ == '__main__':
    tinystory_valid = '/data/TinyStoriesV2-GPT4-valid.txt'
    special_tokens = ['<|endoftext|>']
    vocab_size = 150*(10**3)
    train_bpe(tinystory_valid, vocab_size, special_tokens)