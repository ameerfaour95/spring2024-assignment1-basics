import regex
from tqdm import tqdm
from datetime import datetime
import time
from cs336_basics.utils import GPT2_PRETOKENIZER_PATTERN
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from collections import Counter

def get_pair_freq_table(pertokens_freq: dict):
    pair_freq_table = Counter()
    for pretoken, freq in pertokens_freq.items():
        for pair in zip(pretoken, pretoken[1:]):
            pair_freq_table[pair] = pair_freq_table.get(pair, 0) + freq
    return pair_freq_table

def _merge_and_update_freq_table(
    pertokens_freq: dict,
    pair_freq_table,
    most_freq_pair: tuple[bytes, bytes],
    new_token_bytes: int,
) -> None:

    p0, p1 = most_freq_pair
    new_pertokens_freq = Counter()
    for pretoken, freq in pertokens_freq.items():
        i = 0
        new_pretoken = list(pretoken)
        while i < len(new_pretoken) - 1:
            if new_pretoken[i] == p0 and new_pretoken[i + 1] == p1:
                left  = new_pretoken[i - 1] if i > 0 else None
                right = new_pretoken[i + 2] if i + 2 < len(new_pretoken) else None

                pair_freq_table[most_freq_pair] -= freq
                if left  is not None:
                    pair_freq_table[(left,  p0)] -= freq
                if right is not None:
                    pair_freq_table[(p1, right)] -= freq

                new_pretoken[i : i + 2] = [b"".join([new_token_bytes])]

                if left  is not None:
                    pair_freq_table[(left, new_token_bytes)] = pair_freq_table.get((left, new_token_bytes), 0) + freq
                if right is not None:
                    pair_freq_table[(new_token_bytes, right)] = pair_freq_table.get((new_token_bytes, right), 0) + freq
            
            i += 1
            if i >= len(new_pretoken) - 1:
                token_tuple = tuple(new_pretoken)
                new_pertokens_freq[token_tuple] += freq
    return new_pertokens_freq


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    logging.info("Initializing the vocab")
    vocab = {idx: bytes([idx]) for idx in range(256)}

    for i, special_token in enumerate(special_tokens):
        s_token = special_token.encode("utf-8")
        vocab[256 + i] = s_token

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    for special_token in special_tokens:
        text = text.replace(special_token, "")

    logging.info(f"Pre-tokenizing the text of length {len(text)}")
    pretoknes = Counter(regex.findall(GPT2_PRETOKENIZER_PATTERN, text))

    pertokens_freq = {}
    for pertoken, freq in pretoknes.items():
        bytes_tuple = tuple(bytes([b]) for b in pertoken.encode("utf-8"))
        pertokens_freq[bytes_tuple] = freq

    logging.info("Initializing byte pair frequency table")
    pair_freq_table= get_pair_freq_table(pertokens_freq)

    logging.info("Performing BPE algorithm")
    merges = []
    total_merging = vocab_size - len(vocab)
    with tqdm(total=total_merging, desc="BPE Merges") as pbar:
        while len(vocab) < vocab_size and pair_freq_table:
            most_freq_pair = max(pair_freq_table,
                key=lambda p: (pair_freq_table[p], p))
            merges.append(most_freq_pair)
            new_token_bytes = b"".join(most_freq_pair)
            new_token_id = max(vocab.keys()) + 1
            vocab[new_token_id] = new_token_bytes
            pertokens_freq = _merge_and_update_freq_table(
                pertokens_freq, pair_freq_table, most_freq_pair, new_token_bytes
            )

            pbar.update(1)

    return vocab, merges
