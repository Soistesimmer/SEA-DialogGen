#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main_gen.py --test_data "test_gen.json" \
                                      --model_path "model_gen.pt" \
                                      --output_path "test_gen_seleted.json" \
                                      --batch_size 16 --num_beams 10 \
                                      --num_return_sequences 10
