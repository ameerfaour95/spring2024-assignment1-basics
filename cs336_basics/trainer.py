import argparse, time, math, numpy as np, torch, wandb
from pathlib import Path
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, gradient_clipping, lr_cosine_schedule
from cs336_basics.loss import CrossEntropyLoss
from cs336_basics.data_loader import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.decoding import generate_text
from cs336_basics.tokenizer import Tokenizer


def load_memmap(file_name: str, dtype=np.uint16) -> np.memmap:
    array = np.memmap(file_name, mode="r", dtype=dtype)
    if array.ndim != 1:
        raise ValueError("expect flat 1-D array")
    return array

def write_to_logging(
    generated_text,
    generated_text_path,
    prompt,
    max_tokens,
    temperature,
    top_p
):
    path = Path(generated_text_path)
    file_path = path / "generated_text_during_training.txt"
    if file_path.exists():
        with file_path.open("a") as f:
            f.write("\n")
            f.write(generated_text)
            f.write("\n")
            f.write("="*100)
    else:
        with file_path.open("w") as f:
            f.write(f"The prompt:\n{prompt}\n")
            f.write(
                f"Generation params:\n{max_tokens=}\n{temperature=}\n{top_p=}\n"
            )
            f.write("Generated text during training:\n")
            f.write("-"*100)
            f.write("-"*100)
            f.write("\n")
            f.write(generated_text)
            f.write("\n")
            f.write("="*100)


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-bin", required=True, help="uint16 ids")
    p.add_argument("--valid-bin", help="uint16 ids", default="")
    p.add_argument("--output-dir", default="checkpoints")

    # Architecture
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--d-ff", type=int, default=1536)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--attn-pdrop", type=float, default=None)
    p.add_argument("--residual-pdrop", type=float, default=0.1)

    # Batch & duration
    p.add_argument("--batch", type=int, default=32, help="examples per iteration")
    p.add_argument("--steps", type=int, default=10_000, help="total optimisation iterations")

    # AdamW
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--betas", type=str, default="0.9,0.999")
    p.add_argument("--weight-decay", type=float, default=0.1)

    # LR cosine schedule
    p.add_argument("--lr-min", type=float, default=3e-7)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--cosine-cycle-iters", type=int, default=10000)

    # Gradient clipping
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Checkpointing
    p.add_argument("--resume", action="store_true", help="Resume from checkpoints/output-dir/latest.pt if present")

    # Evaluation / checkpoint cadence (in steps)
    p.add_argument("--eval-every-steps", type=int, default=20)
    p.add_argument("--save-every-steps", type=int, default=100)

    # Device
    p.add_argument("--device", default="cpu")

    # WandB
    p.add_argument("--wandb-project", default=None, help="Weights & Biases project name (enables wandb)")
    p.add_argument("--wandb-run", default=None, help="Optional run name")
    p.add_argument("--wandb-group", default=None, help="Group label for wandb runs")

    # Generation for qualitative checks
    p.add_argument("--generate-prompt", dest="prompt", default=None, help="Prompt used for sample generation during training")
    p.add_argument("--generated-text-path", default=None, help="The generated text that happen during training")

    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--vocab-path", default=None)
    p.add_argument("--merges-path", default=None)
    p.add_argument("--special-tokens", nargs='+', default="<|endoftext|>")
    return p


def eval_loss(model, valid, cfg, loss_fn, device):
    model.eval()
    with torch.no_grad():
        n_tokens = min(20_000, len(valid) - cfg.context_length - 1)
        n_steps = n_tokens // (cfg.batch * cfg.context_length)
        losses = []
        for _ in range(max(1, n_steps)):
            input_batch, target_batch = get_batch(valid, cfg.batch, cfg.context_length, str(device))
            logits = model(input_batch)
            loss = loss_fn(logits, target_batch)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(cfg):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    cross_entropy_fn = CrossEntropyLoss()

    # === WandB ===
    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, group=cfg.wandb_group, config=vars(cfg))

    # === DATA ===
    train_data = load_memmap(cfg.train_bin)
    valid_data = load_memmap(cfg.valid_bin) if cfg.valid_bin else None

    # === MODEL ===
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        attn_pdrop=cfg.attn_pdrop,
        residual_pdrop=cfg.residual_pdrop,
    ).to(device)

    # === TOKENIZER (for sample generation) ===
    tokenizer = Tokenizer.from_files(cfg.vocab_path, cfg.merges_path, [cfg.special_tokens])

    # === OPTIMIZER ===
    beta1, beta2 = (float(b) for b in cfg.betas.split(","))
    opt = AdamW(model.parameters(), lr=cfg.lr, betas=(beta1, beta2), weight_decay=cfg.weight_decay)

    # === CHECKPOINT ===
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(cfg.output_dir) / "latest.pt"
    global_step = 0
    if cfg.resume and checkpoint_path.exists():
        global_step = load_checkpoint(checkpoint_path, model, opt)
        print(f"[resume] step {global_step}")

    # === TRAIN LOOP ===
    n_tokens_per_batch = cfg.batch * cfg.context_length
    print(f"total steps: {cfg.steps} | ~tokens: {cfg.steps * n_tokens_per_batch:,}")

    running_loss = 0.0
    loss_window = 10
    step_start_time = time.time()

    while global_step < cfg.steps:
        # === lr cosine schedule ===
        lr = lr_cosine_schedule(
            global_step,
            max_learning_rate=cfg.lr,
            min_learning_rate=cfg.lr_min,
            warmup_iters=cfg.warmup_iters,
            cosine_cycle_iters=cfg.cosine_cycle_iters,
        )
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        # === get batch ===
        input_batch, target_batch = get_batch(train_data, cfg.batch, cfg.context_length, device=str(device))

        # === forward ===
        logits = model(input_batch)
        loss = cross_entropy_fn(logits, target_batch)
        running_loss += loss.item()

        # === backward ===
        opt.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), cfg.grad_clip)
        opt.step()

        global_step += 1

        # === logging every `loss_window` steps ===
        if global_step % loss_window == 0:
            avg_loss = running_loss / loss_window
            ppl = math.exp(avg_loss)
            elapsed = time.time() - step_start_time
            print(f"Step {global_step}/{cfg.steps} | loss {avg_loss:6.4f} | ppl {ppl:7.2f} | {elapsed:5.1f}s")
            if use_wandb:
                wandb.log({"train/loss": avg_loss, "train/ppl": ppl, "lr": lr, "step": global_step})
            running_loss = 0.0
            step_start_time = time.time()

        # === evaluation ===
        if cfg.valid_bin and (global_step % cfg.eval_every_steps == 0 or global_step == cfg.steps):
            v_loss = eval_loss(model, valid_data, cfg, cross_entropy_fn, device)
            v_ppl = math.exp(v_loss)
            print(f"[val @ step {global_step}] loss {v_loss:6.4f} | ppl {v_ppl:7.2f}")
            if use_wandb:
                wandb.log({"val/loss": v_loss, "val/ppl": v_ppl, "step": global_step})

        # === checkpoint ===
        if global_step % cfg.save_every_steps == 0 or global_step == cfg.steps:
            save_checkpoint(model, opt, global_step, checkpoint_path)
            print(f"[checkpoint] saved {checkpoint_path} @ step {global_step}")
            if use_wandb:
                wandb.save(str(checkpoint_path))

        # === qualitative generation ===
        if cfg.prompt and (global_step % cfg.eval_every_steps == 0):
            sample = generate_text(cfg.prompt, model, tokenizer, cfg.max_tokens, cfg.temperature, cfg.top_p)
            print(f"\nGenerated sample @ step {global_step}:\n{sample}\n")
            if cfg.generated_text_path is not None:
                write_to_logging(
                    sample,
                    cfg.generated_text_path,
                    cfg.prompt,
                    cfg.max_tokens,
                    cfg.temperature,
                    cfg.top_p
                )

    # === final save ===
    save_checkpoint(model, opt, global_step, checkpoint_path)
    print("Training done – final model in", checkpoint_path)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    cfg = build_parser().parse_args()
    train(cfg)
