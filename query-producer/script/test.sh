#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --train_data "train.json" \
                                      --valid_data "valid.json" \
                                      --test_data "test.json" \
                                      --model_path "model_base.pt" \
                                      --output_path "test_selected.json" \
                                      --batch_size 16
