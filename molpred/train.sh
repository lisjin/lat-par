#!/bin/bash
DTAG=ogbg-molhiv
NGPU=2
LR=5e-4
NT_SZ=16
CKPT="ckpt_$(date +%m-%d-%H-%M)"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--ngpu) NGPU=$2; shift ;;
        -l|--lr) LR=$2; shift ;;
        -r|--rnn_sz) NT_SZ=$2; shift ;;
        -c|--ckpt) CKPT=$2; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
python3 train.py --train_data ogbg-molhiv\
    --train_forests "$SCR/k-decomp/${DTAG}/${DTAG}{}_forests_f.hdf5"\
    --train_sep2frags "$SCR/k-decomp/${DTAG}/${DTAG}{}_sep2frags_f.pkl"\
    --nt_size $NT_SZ\
    --concept_dim 72\
    --rel_dim 9\
    --embed_dim 32\
    --ff_embed_dim 32\
    --num_heads 2\
    --graph_layers 2\
    --dropout 0.2\
    --epochs 8\
    --train_batch_size 1600\
    --dev_batch_size 768\
    --lr $LR\
    --weight_decay 3e-6\
    --warmup_steps 900\
    --print_every 300\
    --eval_every 400\
    --ckpt $CKPT\
    --world_size $NGPU\
    --gpus $NGPU\
    --MASTER_ADDR localhost\
    --MASTER_PORT 29500\
    --start_rank 0\
    --num_workers 8
