#!/bin/bash

# baseline
python cs336_basics/train.py \
    --dataset_name='owt' \
    --context_length=1024 \
    --batch_size=128 \
    --vocab_size=50257 \
    --d_model=768 \
    --d_ff=3072 \
    --attn_pdrop=0.0 \
    --resid_pdrop=0.0 \
    --num_layers=12 \
    --num_heads=12 \
    --lr_max=0.0005 \
    --total_iters=10000 \
    --wandb_project='cs336_basics' \
    --wandb_run_name="owt_gpt2" \
    --wandb_logging=True