#%%
import numpy as np
from cs336_basics.tokenizer import Tokenizer
#%%
# -------- TinyStoriesV2-GPT4 --------
vocab_path = "/Users/ameefaour/Desktop/CS336_LLM_from_scratch/spring2024-assignment1-basics/data/output_vocab_TinyStoriesV2-GPT4.txt"
merges_path = "/Users/ameefaour/Desktop/CS336_LLM_from_scratch/spring2024-assignment1-basics/data/output_merges_TinyStoriesV2-GPT4.txt"
#%%
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(
    vocab_path, merges_path, special_tokens
)
# %%
data_path = "/Users/ameefaour/Desktop/CS336_LLM_from_scratch/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
with open(data_path, "r") as file:
    text = file.read()
#%%
tokens_ids = tokenizer.encode(text)
tokens_ids_np = np.array(tokens_ids, dtype=np.uint16)
np.save("/Users/ameefaour/Desktop/CS336_LLM_from_scratch/spring2024-assignment1-basics/data/tokenized_data/encoded_tokens_TinyStoriesV2-GPT4.npy", tokens_ids_np)
# %%
# -------- OpenWebText --------
# vocab_path = "/Users/ameefaour/Desktop/CS336_LLM_from_scratch/spring2024-assignment1-basics/data/output_vocab_owt.txt"
# merges_path = "/Users/ameefaour/Desktop/CS336_LLM_from_scratch/spring2024-assignment1-basics/data/output_merges_owt.txt"
# #%%
# special_tokens = ["<|endoftext|>"]
# tokenizer = Tokenizer.from_files(
#     vocab_path, merges_path, special_tokens
# )
# #%%
# data_path = "/Users/ameefaour/Desktop/CS336_LLM_from_scratch/spring2024-assignment1-basics/data/owt_train.txt"
# with open(data_path, "r") as file:
#     text = file.read()
# #%%
# tokens_ids = tokenizer.encode(text)
# tokens_ids_np = np.array(tokens_ids, dtype=np.uint16)
# np.save("/Users/ameefaour/Desktop/CS336_LLM_from_scratch/spring2024-assignment1-basics/data/tokenized_data/encoded_tokens_owt.npy", tokens_ids_np)
# # %%