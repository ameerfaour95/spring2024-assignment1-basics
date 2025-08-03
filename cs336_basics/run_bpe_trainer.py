from cs336_basics.train_bpe import train_bpe
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, help="provide the txt file path")
    parser.add_argument("--vocab-size", type=int, required=True, help="the vocab size output")
    parser.add_argument("--special-tokens", type=str, nargs="+", required=True, help="pass the special tokens list")
    parser.add_argument("--output-vocab", required=True, help="the output txt where to save vocab")
    parser.add_argument("--output-merges", required=True, help="the output txt where to save merges")
    args = parser.parse_args()

    logging.info("*** Starting BPE training ***")

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    logging.info("*** Finished BPE training ***")
    
    with open(args.output_merges, "w") as f:
        f.write(str(merges))

    with open(args.output_vocab, "w") as f:
        f.write(str(vocab))

if __name__ == "__main__":
    main()