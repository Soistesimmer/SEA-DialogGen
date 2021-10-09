#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_gen.py --train_data "train_gen.json" \
                                      --valid_data "valid_gen.json" \
                                      --test_data "test_gen.json" \
                                      --model_path "model_gen.pt" \
                                      --epoch 10 --batch_size 4 \
                                      --training --early_stop 3 \
                                      --gradient_accumulate 16 \
                                      --lr 1e-5 --beg_rl 3 \
                                      --beg_val -1

#CUDA_VISIBLE_DEVICES=0 python main_gen.py --train_data "train_gen.json" \
#                                      --valid_data "valid_gen.json" \
#                                      --test_data "test_gen.json" \
#                                      --model_path "model_gen.pt" \
#                                      --epoch 20 --batch_size 4 \
#                                      --training --early_stop 3 \
#                                      --gradient_accumulate 16 \
#                                      --lr 1e-5 --beg_rl 3
