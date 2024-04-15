from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser

from cs336_basics.tokenizer import Tokenizer


@dataclass
class SampleConfig:
    # tokenizer parameters
    vocab_filepath: Optional[str] = field(default='data/out/tinystories_vocab.json')
    merges_filepath: Optional[str] = field(default='data/out/tinystories_merges.txt')
    special_tokens: Optional[Iterable[str]] = field(default_factory=lambda: ['<|endoftext|>'])


# loading tokenizer
tokenizer = Tokenizer.from_files(**asdict(config))
# print(tokenizer.decode(x[0].cpu().numpy()))

