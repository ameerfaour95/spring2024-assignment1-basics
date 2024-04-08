import pickle

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def serialize(obj, path: str):
    serialized_obj = pickle.dumps(obj)
    with open(path, 'wb') as f:
        f.write(serialized_obj)

def deserialize(path: str):
    with open(path, 'rb') as f:
        serialized_obj = pickle.load(f)
    return serialized_obj