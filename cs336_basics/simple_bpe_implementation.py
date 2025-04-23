#%%
def get_max_occurrence(ids):
    count_pair = {}
    for token_1, token_2 in zip(ids[:], ids[1:]):
        pair = (token_1,token_2)
        count_pair[pair] = count_pair.get(pair, 0) + 1
    return {
        k: v for k, v in sorted(
              count_pair.items(), key=lambda item: item[1], reverse=True
            )
        }
def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i]==pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids

def BPE_train(ids, num_of_merges):
    merges = {}
    for i in range(num_of_merges):
        occurrence_dict = get_max_occurrence(ids)
        max_occurrence_pair = max(occurrence_dict, key=occurrence_dict.get)
        idx = 256 + i
        print(f'Merging {max_occurrence_pair} into new token {idx}')
        ids = merge(ids, max_occurrence_pair, idx)
        merges[max_occurrence_pair] = idx
    return ids, merges

#%%
corpus = """low low low low
lower lower widest widest widest
newest newest newest newest newest"""

special_token = "<|endoftext|>"

tokens = corpus.encode("utf-8")
tokens = list(map(int, tokens)) 
print('---')
print(corpus)
print(f"length: {len(corpus)}")
print('---')
print(tokens)
print(f"length: {len(tokens)}")
#%%
ids, merges = BPE_train(tokens, 6)
# %%
print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens)/len(ids):.2f}X")
# %%
vocab = {idx: bytes([idx]) for idx in range(0, 256)}
for (p0, p1), idx in merges.items():
    print(p0, p1, idx)
    vocab[idx] = vocab[p0] + vocab[p1]
# %%
vocab