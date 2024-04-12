from cs336_basics.train_bpe import train_bpe
from cs336_basics.utils.io import serialize, deserialize
import cProfile
import wandb


# configs variables
wandb_name = 'cs336_basics'
wandb_run_name = 'train_bpe_expts_owt'
special_tokens = ['<|endoftext|>']
text_source = 'data/owt_train.txt'
vocab_size = 32*(10**3)
output_vocab_path = 'data/out/owt_vocab.pkl'
output_merge_path = 'data/out/owt_merges.pkl'
num_worker = 1
config = dict(wandb_name=wandb_name, wandb_run_name=wandb_run_name,
              special_tokens=special_tokens, text_source=text_source,
              vocab_size=vocab_size, output_vocab_path=output_vocab_path,
              output_merge_path=output_merge_path)

# wandb logging
# wandb.init(project=wandb_name, name=wandb_run_name, config=config)


# Training BPE
pr = cProfile.Profile()
pr.enable()
vocab, merges = train_bpe(text_source, vocab_size, special_tokens, progress_bar=True, num_worker=num_worker)
pr.disable()

# Print time taken in units of hours
pr.print_stats(sort='time')

# Serialize and save the vocab and merges
serialize(vocab, output_vocab_path)
serialize(merges, output_merge_path)