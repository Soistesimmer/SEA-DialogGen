import json
import random

import torch

from torch.utils import data
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

from utils import load_json_file, flat_list

'''
example format as follow
{
    'topic': {}
    'dialog': [{
        'text': 'text',
        'candidate': [{'query'{
            'wiki_passages': [],
            'tokens': tokens,
            'score': score, 
        }},]
    },]
}
'''
class DialogSet(data.Dataset):
    def __init__(self, args, data_path, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.training = args.training
        self.max_length = args.max_cx_len
        self.max_pos_emb_size = args.max_pos_emb_size
        self.dataset = load_json_file(data_path)
        self.additional_special_tokens = args.additional_special_tokens
        self.instance = self.get_training_instance(self.dataset)


    # ensure input length less than self.max_length
    def get_training_instance(self, data):
        instances = []
        hit_cnt, ins_cnt = 0, 0

        def lower_set(x):
            return set([y.lower() for y in x])

        for session_id, example in enumerate(data):
            for turn_id, turn in enumerate(example['dialog']):
                if turn['speaker'] == 'wizard' and turn_id != 0:
                    ins_cnt += 1
                    candidates = []
                    if 'candidates' in turn and 'target_query' in turn:
                        candidates = [list(candidate.keys())[0] for candidate in turn['candidates']]
                        if set(candidates) & set(turn['target_query']):
                            hit_cnt += 1
                    if self.training and ('candidates' not in turn or len(lower_set(candidates)) < 2):
                        continue
                    instances.append((session_id, turn_id))
        print(hit_cnt / ins_cnt, hit_cnt, ins_cnt)
        return instances

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, index):
        session_id, turn_id = self.instance[index]
        bos_token_id, eos_token_id = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        emt_token_id = self.additional_special_tokens['<emt>']
        dialog = self.dataset[session_id]['dialog']

        source_token_ids = flat_list([[self.additional_special_tokens['<{}>'.format(turn['speaker'])]]
                                        + turn['tokens'] for turn in dialog[:turn_id]])
        source_token_ids = source_token_ids[-self.max_length:]

        if self.training:
            target_token_ids_list = []
            reward_list = []
            query_list = []
            if 'candidates' in dialog[turn_id]:
                for candidate in dialog[turn_id]['candidates']:
                    k, v = list(candidate.items())[0]
                    k = k.lower()
                    if k in query_list:
                        continue
                    query_list.append(k)
                    target_token_ids = [bos_token_id] + v['tokens'] + [eos_token_id]
                    target_token_ids_list.append(target_token_ids)
                    reward_list.append(v['score'] if isinstance(v['score'], float) or isinstance(v['score'], int) else v['score'][0])
            target_token_ids = [bos_token_id] + [emt_token_id] + [eos_token_id]
            target_token_ids_list.append(target_token_ids)

            reward_list.append(min(reward_list) if reward_list else 0)
            reward_list = torch.tensor(reward_list)
            min_reward, max_reward = reward_list.min(), reward_list.max()
            if min_reward == max_reward:
                reward_list[:] = 0
            else:
                reward_list = (reward_list - min_reward) / (max_reward - min_reward) - 0.5
            reward_list = reward_list.tolist()
            if 'target_query' in dialog[turn_id]:
                targets = [x.lower() for x in dialog[turn_id]['target_query']]
                target_indices = [query_list.index(x) for x in targets if x in query_list]
            else:
                target_indices = []

            return source_token_ids, target_token_ids_list, reward_list, target_indices
        else:
            if 'target_query' in dialog[turn_id]:
                targets = [x.lower() for x in dialog[turn_id]['target_query']]
                query_list = []
                for candidate in dialog[turn_id]['candidates']:
                    k = list(candidate.keys())[0]
                    k = k.lower()
                    if k in query_list:
                        continue
                    query_list.append(k)
                target_indices = [query_list.index(x.lower()) for x in targets if x.lower() in query_list]
            else:
                targets = []
                target_indices = []
            return source_token_ids, targets, target_indices
