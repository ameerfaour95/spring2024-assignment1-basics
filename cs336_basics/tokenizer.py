import regex as re
from typing import Dict, Tuple, Iterable, List
from cs336_basics.utils.io import get_tokenizer_from_vocab_merges_path, GPT2_PRETOKENIZER_PATTERN
import concurrent.futures

def get_pairs(ids: Iterable[int]) -> Iterable[Tuple[int, int]]:
    """ Return a set of pairs in int ids """
    pairs = set()
    for pair in zip(ids, ids[1:]):
        pairs.add(pair)
    return pairs

def update(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """ Update the ids by merging the pairs """
    new_ids = []
    i = 0
    while i < len(ids):
        curr_pair = tuple(ids[i:i+2])
        if curr_pair == pair:
            new_ids.append(new_id)
            i += 1
        else:
            new_ids.append(ids[i])
        i += 1
    return new_ids

def _fix_vocab(vocab_i_to_b: Dict[int, bytes], vocab_b_to_i: Dict[str, bytes]):
    """ Make sure all bytes are in the vocab """
    for i in range(256):
        byte = bytes([i])
        if byte not in vocab_b_to_i:
            vocab_b_to_i[byte] = len(vocab_b_to_i)
            vocab_i_to_b[len(vocab_i_to_b)] = byte
    return dict(int_to_byte=vocab_i_to_b, byte_to_int=vocab_b_to_i)

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: Iterable[Tuple[bytes, bytes]], special_tokens: Iterable[str]=None):
        self.vocab = {}
        self.vocab['int_to_byte'] = vocab
        self.vocab['byte_to_int'] = {v: k for k, v in vocab.items()}
        self.vocab = _fix_vocab(self.vocab['int_to_byte'], self.vocab['byte_to_int'])

        # reorganzie merges into pair -> new token id dict
        self.merges = {}
        for a, b in merges:
            id_pair = (self.vocab['byte_to_int'][a], self.vocab['byte_to_int'][b])
            self.merges[id_pair] = self.vocab['byte_to_int'][a+b]
        
        # add special tokens as string to id mapping
        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode("utf-8")
                if token_byte not in self.vocab['byte_to_int']:
                    self.vocab['byte_to_int'][token_byte] = len(self.vocab['byte_to_int'])
                    self.vocab['int_to_byte'][len(self.vocab['int_to_byte'])] = token_byte
                    self.special_tokens[token] = len(self.vocab['int_to_byte'])
                else:
                    self.special_tokens[token] = self.vocab['byte_to_int'][token_byte]
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, **kwargs):
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    @property
    def vocab_size(self):
        return len(self.vocab['int_to_byte'])

    def _encode_chunk(self, chunk: str) -> List[int]:
        """
        Encode a short chunk of text with no special tokens.
        """
        text_bytes = chunk.encode("utf-8")
        ids = [self.vocab['byte_to_int'][bytes([b])] for b in chunk.encode("utf-8")]
        while len(ids)>=2:
            pairs = get_pairs(ids)
            high_priority_pair = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            if high_priority_pair not in self.merges:
                break
            new_id = self.merges[high_priority_pair]
            ids = update(ids, high_priority_pair, new_id)
        return ids
    
    def _encode_no_special(self, text: str, num_worker) -> List[int]:
        """
        Encode the text without special tokens.
        """
        text_chunks = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
            ids = list(executor.map(self._encode_chunk, text_chunks))
        return sum(ids, [])


    def encode(self, text: str, num_worker=10) -> List[int]:
        """
        Encode the text into a list of token ids.
        """
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            special_chunks = re.split(special_pattern, text)
        else:
            special_chunks = [text]
        ids = []
        for chunk in special_chunks:
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids += self._encode_no_special(chunk, num_worker=num_worker)
        return ids
    
    def encode_iterable(self, texts: Iterable[str], num_worker=1) -> Iterable[List[int]]:
        """
        Encode the texts into a list of token ids.
        """
        for text in texts:
            ids = self.encode(text, num_worker=num_worker)
            for id in ids:
                yield id

    def decode(self, ids: List[int]) -> str:
        """
        Decode the token ids into the original text.
        """
        text_bytes = b''.join([self.vocab['int_to_byte'][i] for i in ids])
        return text_bytes.decode("utf-8", errors="replace")