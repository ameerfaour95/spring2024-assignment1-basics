import regex
from collections.abc import Iterable
from cs336_basics.utils import GPT2_PRETOKENIZER_PATTERN

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.token_to_ids = {v:k for k,v in vocab.items()}
        self.merges = merges
        self.bpe_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        if self.special_tokens:
            self.special_pattern = "(" + "|".join(regex.escape(s) for s in self.special_tokens) + ")"
            
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        assert vocab_filepath and merges_filepath, \
            "Must provide vocab_filepath and merges_filepath"

        if special_tokens:
            assert isinstance(special_tokens, list), \
                "Special tokens must passed in a list format"

        if vocab_filepath.endswith("txt"):
            with open(vocab_filepath, "r") as f:
                vocab_text = f.read()
            if vocab_text:
                try:
                    vocab_raw = eval(vocab_text)
                    vocab = {i: (v if isinstance(v, bytes) else v.encode()) for i, v in vocab_raw.items()}
                except:
                    raise ValueError("Couldnt parse vocab")

        elif vocab_filepath.endswith("json"):
            raise NotImplementedError

        if merges_filepath.endswith("txt"):
            with open(merges_filepath, "r") as f:
                merges_text = f.read()
            if merges_text:
                try:
                    merges_raw = eval(merges_text)
                    merges = [
                        (a if isinstance(a, bytes) else a.encode(), b if isinstance(b, bytes) else b.encode())
                        for a, b in merges_raw
                    ]
                except:
                    raise ValueError("Couldnt parse merges")
    
        elif merges_filepath.endswith("json"):
            raise NotImplementedError

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens
        )

    def _encode_word(self, word: str) -> list[bytes]:
        tokens = [bytes([b]) for b in word.encode("utf-8")]

        while len(tokens) > 1:
            min_rank = float("inf")
            min_pos = None
            for i, (p0, p1) in enumerate(zip(tokens, tokens[1:])):
                rank = self.bpe_rank.get((p0, p1), float("inf"))
                if rank < min_rank:
                    min_rank, min_pos = rank, i
            if min_pos is None:
                break

            best_pair = tuple(tokens[min_pos:min_pos + 2])
    
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == best_pair:
                    tokens[i:i + 2] = [tokens[i] + tokens[i + 1]]
                else:
                    i += 1
        return tokens

    def encode(self, text: str) -> list[bytes]:
        if self.special_tokens:
            chunks = regex.split(self.special_pattern, text)
        else:
            chunks = [text]
        
        pre_tokens = []

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                pre_tokens.append([chunk.encode("utf-8")])
                continue
            
            words = regex.findall(GPT2_PRETOKENIZER_PATTERN, chunk)

            for w in words:
                pre_tokens.append(self._encode_word(w))

        return [self.token_to_ids[b]
                for tok in pre_tokens
                for b in tok]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
            if not isinstance(iterable, Iterable):
                raise TypeError("Expected an iterable of strings")
            for part in iterable:
                for i in self.encode(part):
                    yield i
        
    def decode(self, ids: list[int]) -> str:
        tokens = b"".join([self.vocab[id] for id in ids])
        return tokens.decode("utf-8", errors="replace")