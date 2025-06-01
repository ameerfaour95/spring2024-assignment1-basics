import regex
import time
from tqdm import tqdm
from datetime import datetime
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

    start_wall = datetime.now()
    start_perf = time.perf_counter()
    print(f"[{start_wall:%Y‑%m‑%d %H:%M:%S}]▶ loading input")
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    end_wall   = datetime.now()
    duration_s = time.perf_counter() - start_perf
    print(f"[{end_wall:%Y‑%m‑%d %H:%M:%S}] *** Finished loading input "
          f"in {duration_s:.2f} s ***")

    for special_token in special_tokens:
        text = text.replace(special_token, "")

    start_wall = datetime.now()
    start_perf = time.perf_counter()
    print(f"[{start_wall:%Y‑%m‑%d %H:%M:%S}]▶ splitting words")
    words = regex.findall(GPT2_PRETOKENIZER_PATTERN, text)

    end_wall   = datetime.now()
    duration_s = time.perf_counter() - start_perf
    print(f"[{end_wall:%Y‑%m‑%d %H:%M:%S}] *** Finished splitting words "
          f"in {duration_s:.2f} s ***")

    start_wall = datetime.now()
    start_perf = time.perf_counter()
    print(f"[{start_wall:%Y‑%m‑%d %H:%M:%S}]▶ encoding")
    ids = [list(w.encode("utf-8")) for w in words]

    end_wall   = datetime.now()
    duration_s = time.perf_counter() - start_perf
    print(f"[{end_wall:%Y‑%m‑%d %H:%M:%S}] *** Finished encoding "
          f"in {duration_s:.2f} s ***")

    del words

    start_wall = datetime.now()
    start_perf = time.perf_counter()
    print(f"[{start_wall:%Y‑%m‑%d %H:%M:%S}]▶ init vocab")
    vocab = {idx: bytes([idx]) for idx in range(256)}

    end_wall   = datetime.now()
    duration_s = time.perf_counter() - start_perf
    print(f"[{end_wall:%Y‑%m‑%d %H:%M:%S}] *** Finished init vocab "
          f"in {duration_s:.2f} s ***")

    start_wall = datetime.now()
    start_perf = time.perf_counter()
    print(f"[{start_wall:%Y‑%m‑%d %H:%M:%S}]▶ handling special tokens")
    for i, special_token in enumerate(special_tokens):
        s_token = special_token.encode("utf-8")
        vocab[256 + i] = s_token

    start_wall = datetime.now()
    start_perf = time.perf_counter()
    print(f"[{start_wall:%Y‑%m‑%d %H:%M:%S}]▶ init freq table")
    pair_freq_table, pair_index = get_stats(ids)

    start_wall = datetime.now()
    start_perf = time.perf_counter()
    print(f"[{start_wall:%Y‑%m‑%d %H:%M:%S}]▶ merging & updating freq table")
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


    end_wall   = datetime.now()
    duration_s = time.perf_counter() - start_perf
    print(f"[{end_wall:%Y‑%m‑%d %H:%M:%S}] *** Finished merging & updating freq table "
          f"in {duration_s:.2f} s ***")
    return vocab, merges