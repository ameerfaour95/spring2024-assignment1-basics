#!/bin/bash

# Define the range of batch sizes
batch_sizes=(1 2 4 8 16 32 64 128 256 512 1024)

# Loop through each batch size
for batch_size in "${batch_sizes[@]}"
do
    echo "Running training with batch size: $batch_size"
    python cs336_basics/train.py \
        --dataset_name='tinystories' \
        --context_length=256 \
        --batch_size=$batch_size \
        --vocab_size=10000 \
        --d_model=512 \
        --d_ff=2048 \
        --attn_pdrop=0.0 \
        --resid_pdrop=0.0 \
        --num_layers=4 \
        --num_heads=16 \
        --lr_max=0.001 \
        --total_iters=10000 \
        --wandb_project='cs336_basics' \
        --wandb_run_name="tinystories_batchsize_${batch_size}" \
        --wandb_logging=False \
        --eval_iters=1
done
