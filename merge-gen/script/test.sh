#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py --test_data "test_fid_20_1.json" \
                                      --model_path "model.pt" \
                                      --output_path "predictions.json" \
                                      --batch_size 16 --num_beams 4 \
                                      --max_kn_len 256 --topk 5 \
                                      --type "pred_bm25"
#
#CUDA_VISIBLE_DEVICES=1 python main.py --test_data "test_fid_20_1.json" \
#                                      --model_path "model.pt" \
#                                      --output_path "predictions.json" \
#                                      --batch_size 16 --num_beams 4 \
#                                      --max_kn_len 256 --topk 5 \
#                                      --type "pred_bm25" --ppl
