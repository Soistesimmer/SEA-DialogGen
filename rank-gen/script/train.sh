#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py --train_data "train_d20_q1_4p.json" \
                                      --valid_data "valid_d20_q1_4p.json" \
                                      --model_path "model.pt" \
                                      --epoch 20 --batch_size 8 \
                                      --training --early_stop 3 \
                                      --gradient_accumulate 8 \
                                      --max_kn_len 256 \
                                      --rank_gpu_ids 1 \
                                      --gen_gpu_ids 0 \
                                      --lr 1e-5 --beg_rl 0 \
                                      --alpha 2 --topk 5 \
                                      --type "pred_bm25"

