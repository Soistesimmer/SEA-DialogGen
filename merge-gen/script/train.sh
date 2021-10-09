#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python main.py --train_data "train_fid_20_1.json" \
                                      --valid_data "valid_fid_20_1.json" \
                                      --model_path "model.pt" \
                                      --epoch 20 --batch_size 8 \
                                      --training --early_stop 3 \
                                      --gradient_accumulate 8 \
                                      --max_kn_len 256 --dp \
                                      --lr 1e-5 --topk 5 \
                                      --type "pred_bm25"
