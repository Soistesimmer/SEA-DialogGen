#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python main.py --test_data "test_d20_q1_4p.json" \
                                      --model_path "model_np.pt" \
                                      --output_path "predictions.json" \
                                      --batch_size 16 --num_beams 4 --max_kn_len 256 \
                                      --rank_gpu_ids 0 --gen_gpu_ids 0 --topk 20 \
                                      --type "bm25"

CUDA_VISIBLE_DEVICES=4 python main.py --test_data "test_d20_q1_4p.json" \
                                      --model_path "model_np.pt" \
                                      --output_path "predictions.json" \
                                      --batch_size 16 --num_beams 4 --max_kn_len 256 \
                                      --rank_gpu_ids 0 --gen_gpu_ids 0 --topk 20 \
                                      --type "bm25" --ppl