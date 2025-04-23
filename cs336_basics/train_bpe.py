import regex
from tqdm import tqdm
from cs336_basics.utils import GPT2_PRETOKENIZER_PATTERN

def get_stats(tokens: list[list[bytes]]) -> dict[tuple[bytes, bytes], int]:
    stats = {}
    for token in tokens:
        for pair in zip(token[:], token[1:]):
            stats[pair] = stats.get(pair, 0) + 1
    return stats

def _merge_and_update_freq_table(
    pair_freq_table: dict[tuple[bytes, bytes], int],
    ids: list[list[bytes]],
    most_freq_pair: tuple[bytes, bytes],
    new_token: bytes,
) -> None:

    p0, p1 = most_freq_pair

    for tok in ids:
        i = 0
        while i < len(tok) - 1:
            if tok[i] != p0 or tok[i + 1] != p1:
                i += 1
                continue

            left  = tok[i - 1] if i > 0 else None
            right = tok[i + 2] if i + 2 < len(tok) else None

            pair_freq_table[most_freq_pair] -= 1
            if left  is not None: pair_freq_table[(left,  p0)] -= 1
            if right is not None: pair_freq_table[(p1, right)] -= 1

            tok[i : i + 2] = [new_token]

            if left  is not None:
                pair_freq_table[(left, new_token)] = pair_freq_table.get((left, new_token), 0) + 1
            if right is not None:
                pair_freq_table[(new_token, right)] = pair_freq_table.get((new_token, right), 0) + 1

            if i: i -= 1

    for k in [k for k, v in pair_freq_table.items() if v <= 0]:
        del pair_freq_table[k]

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    print("▶ loading input")
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    for special_token in special_tokens:
        text = text.replace(special_token, "")

    print("▶ splitting")
    words = regex.findall(GPT2_PRETOKENIZER_PATTERN, text)

    print("▶ encoding")
    ids = [[bytes([b]) for b in w.encode("utf-8")] for w in words]

    print("▶ init vocab")
    vocab = {idx: bytes([idx]) for idx in range(256)}

    print("▶ handling special tokens")
    for i, special_token in enumerate(special_tokens):
        s_token = special_token.encode("utf-8")
        vocab[256 + i] = s_token
    
    print("▶ init freq table")
    pair_freq_table = get_stats(ids)

    print("▶ start training")
    merges = []
    total_merging = vocab_size - len(vocab)
    with tqdm(total=total_merging, desc="BPE Merges") as pbar:
        while len(vocab) < vocab_size and pair_freq_table:
            most_freq_pair = max(pair_freq_table,
                            key=lambda p: (pair_freq_table[p], p))
            new_token = b"".join(most_freq_pair)
            _merge_and_update_freq_table(pair_freq_table, ids, most_freq_pair, new_token)
            merges.append(most_freq_pair)
            vocab[max(vocab) + 1] = new_token
            pbar.update(1)

    return vocab, merges
