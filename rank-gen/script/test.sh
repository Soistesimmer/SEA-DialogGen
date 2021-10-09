#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=4 python main.py --test_data "test_d20_q1_4p.json" \
                                      --model_path "model.pt" \
                                      --output_path "predictions.json" \
                                      --batch_size 16 --num_beams 4 \
                                      --rank_gpu_ids 0 --gen_gpu_ids 0 \
                                      --type "pred_bm25" --topk 20

#CUDA_VISIBLE_DEVICES=4 python main.py --test_data "test_d20_q1_4p.json" \
#                                      --model_path "model.pt" \
#                                      --output_path "predictions.json" \
#                                      --batch_size 16 --num_beams 4 \
#                                      --rank_gpu_ids 0 --gen_gpu_ids 0 \
#                                      --type "pred_bm25" --topk 20 --ppl
