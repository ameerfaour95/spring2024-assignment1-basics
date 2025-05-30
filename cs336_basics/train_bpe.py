import regex
from tqdm import tqdm
from cs336_basics.utils import GPT2_PRETOKENIZER_PATTERN

def get_stats(ids):
    pair_freq_table = {}
    pair_index = {}
    for sent_idx, token in enumerate(ids):
        for i, pair in enumerate(zip(token, token[1:])):
            pair_freq_table[pair] = pair_freq_table.get(pair, 0) + 1
            pair_index.setdefault(pair, {}).setdefault(sent_idx, []).append(i)
    return pair_freq_table, pair_index

def _merge_and_update_freq_table(
    pair_freq_table,
    pair_index,
    ids: list[list[int]],
    most_freq_pair: tuple[bytes, bytes],
    new_token_id: int,
) -> None:
    p0, p1 = most_freq_pair
    occurrences = pair_index.pop(most_freq_pair)
    for sent_idx in occurrences.keys():
        tok = ids[sent_idx]
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

            tok[i : i + 2] = [new_token_id]

            if left  is not None:
                pair_freq_table[(left, new_token_id)] = pair_freq_table.get((left, new_token_id), 0) + 1
                pair_index.setdefault((left, new_token_id),{}).setdefault(sent_idx, []).append(i)
            if right is not None:
                pair_freq_table[(new_token_id, right)] = pair_freq_table.get((new_token_id, right), 0) + 1
                pair_index.setdefault((new_token_id, right),{}).setdefault(sent_idx, []).append(i)
            if i: i -= 1

    for k in [k for k, v in pair_freq_table.items() if v <= 0]:
        pair_freq_table.pop(k, None)
        pair_index.pop(k, None)

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
    ids = [list(w.encode("utf-8")) for w in words]

    print("▶ deleting 'words' var")
    del words

    print("▶ init vocab")
    vocab = {idx: bytes([idx]) for idx in range(256)}

    print("▶ handling special tokens")
    for i, special_token in enumerate(special_tokens):
        s_token = special_token.encode("utf-8")
        vocab[256 + i] = s_token
    
    print("▶ init freq table")
    pair_freq_table, pair_index = get_stats(ids)

    print("▶ start training")
    merges = []
    total_merging = vocab_size - len(vocab)
    with tqdm(total=total_merging, desc="BPE Merges") as pbar:
        while len(vocab) < vocab_size and pair_freq_table:
            most_freq_pair = max(pair_freq_table,
                key=lambda p: (pair_freq_table[p], (vocab[p[0]], vocab[p[1]])))
            p0, p1 = most_freq_pair
            new_token_bytes = vocab[p0] + vocab[p1]
            new_token_id = max(vocab) + 1
            _merge_and_update_freq_table(
                pair_freq_table, pair_index, ids, most_freq_pair, new_token_id
            )
            merges.append((vocab[p0] ,vocab[p1]))
            vocab[max(vocab) + 1] = new_token_bytes
            pbar.update(1)

    return vocab, merges