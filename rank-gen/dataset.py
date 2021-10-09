import json
import random

import torch

from torch.utils import data
import numpy as np
from transformers import AutoTokenizer

from utils import load_json_file, flat_list


class DialogSet(data.Dataset):
    def __init__(self, data_path, tokenizer, args):
        self.args = args
        self.type = args.type
        self.topk = args.topk
        self.max_cx_len = args.max_cx_len
        self.max_kn_len = args.max_kn_len
        self.max_length = args.max_length
        self.tokenizer = tokenizer
        self.dataset = load_json_file(data_path)
        self.additional_special_tokens = args.additional_special_tokens
        self.instance = self.get_training_instance(self.dataset)

    # ensure input length less than self.max_length
    def get_training_instance(self, data):
        instances = []
        for session_id, example in enumerate(data):
            for turn_id, turn in enumerate(example['dialog']):
                if turn['speaker'] == 'wizard' and turn_id != 0:
                    instances.append((session_id, turn_id))
        return instances

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, index):
        session_id, turn_id = self.instance[index]
        bos_token_id, eos_token_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        cls_token_id = self.tokenizer.cls_token_id
        example = self.dataset[session_id]
        dialog = example['dialog']
        history_token_ids = [cls_token_id] + flat_list([[self.additional_special_tokens['<{}>'.format(turn['speaker'])]]
                             + turn['tokens'] for turn in dialog[:turn_id]])[-self.max_cx_len+1:]
        passages = [[cls_token_id, self.additional_special_tokens['<e>']]]
        if self.type in dialog[turn_id]:
            for passage in dialog[turn_id][self.type][:self.topk]:
                passages.append([cls_token_id]+passage['tokens'][:self.max_kn_len-1])

        response_token_ids = ([bos_token_id] + dialog[turn_id]['tokens']+[eos_token_id])[:self.max_length]

        return history_token_ids, passages, response_token_ids
