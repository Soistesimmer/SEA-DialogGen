#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py --train_data "train.json" \
                                      --valid_data "valid.json" \
                                      --test_data "test.json" \
                                      --model_path "model_base.pt" \
                                      --epoch 30 --batch_size 16 \
                                      --training --early_stop 5 \
                                      --gradient_accumulate 16 \
                                      --lr 1e-5 --beg_rl 1
