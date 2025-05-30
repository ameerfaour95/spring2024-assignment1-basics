from cs336_basics.train_bpe import train_bpe
import argparse
from datetime import datetime
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, help="provide the txt file path")
    parser.add_argument("--vocab-size", type=int, required=True, help="the vocab size output")
    parser.add_argument("--special-tokens", type=str, nargs="+", required=True, help="pass the special tokens list")
    parser.add_argument("--output-vocab", required=True, help="the output txt where to save vocab")
    parser.add_argument("--output-merges", required=True, help="the output txt where to save merges")
    args = parser.parse_args()

    start_wall = datetime.now()
    start_perf = time.perf_counter()

    print(f"[{start_wall:%Y‑%m‑%d %H:%M:%S}] *** Starting BPE training ***")

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    end_wall   = datetime.now()
    duration_s = time.perf_counter() - start_perf

    print(f"[{end_wall:%Y‑%m‑%d %H:%M:%S}] *** Finished BPE training "
          f"in {duration_s:.2f} s ***")

    
    with open(args.output_merges, "w") as f:
        f.write("The merges:\n\n")
        f.write(str(merges))

    with open(args.output_vocab, "w") as f:
        f.write("The vocab results:\n\n")
        f.write(str(vocab))

if __name__ == "__main__":
    main()