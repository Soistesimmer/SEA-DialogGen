#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py --test_data "test_fid_20_1.json" \
                                      --model_path "model_np.pt" \
                                      --output_path "predictions_np.json" \
                                      --batch_size 4 --num_beams 4 \
                                      --max_kn_len 256 --topk 20 \
                                      --type "bm25"

#CUDA_VISIBLE_DEVICES=0 python main.py --test_data "test_fid_20_1.json" \
#                                      --model_path "model_np.pt" \
#                                      --output_path "predictions_np.json" \
#                                      --batch_size 4 --num_beams 4 \
#                                      --max_kn_len 256 --topk 20 \
#                                      --type "bm25" --ppl
