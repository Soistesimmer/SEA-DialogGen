import json
import random
import re

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
                        if set(candidates)&set(turn['target_query']):
                            hit_cnt += 1
                    if self.training and ('candidates' not in turn or len(lower_set(candidates))<2):
                        continue
                    instances.append((session_id, turn_id))
        print(hit_cnt/ins_cnt, hit_cnt, ins_cnt)
        return instances

    def __len__(self):
        return len(self.instance)

    def find_sublist(self, query, main_list):
        found_list = []
        j = len(query)
        for i, s in enumerate(main_list):
            if s == query[0] and query == main_list[i:i+j]:
                found_list.append((i, i+j))
        return found_list

    def __getitem__(self, index):
        session_id, turn_id = self.instance[index]
        cls_token_id, sep_token_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        dialog = self.dataset[session_id]['dialog']

        source_token_ids = [[self.additional_special_tokens['<{}>'.format(turn['speaker'])]] + turn['tokens'] + [sep_token_id] for turn in dialog[:turn_id]]
        source_token_type_ids = [[0]*len(x) if x[0] == self.additional_special_tokens['<wizard>'] else [1]*len(x) for x in source_token_ids]
        source_token_ids = [cls_token_id] + flat_list(source_token_ids)[-self.max_length+1:]
        source_token_type_ids = [0] + flat_list(source_token_type_ids)[-self.max_length+1:]

        candidate_list = []
        query_list = []
        reward_list = []
        if 'candidates' in dialog[turn_id]:
            for candidate in dialog[turn_id]['candidates']:
                k, v = list(candidate.items())[0]
                k = k.lower()
                if k in query_list:
                    continue
                found_list = self.find_sublist(v['tokens'], source_token_ids)
                if found_list and k not in query_list:
                    query_list.append(k)
                    candidate_list.append(found_list)
                    reward_list.append(v['score'])
            reward_list = torch.tensor(reward_list)
            # 4 bl
            if len(reward_list.shape) > 1:
                reward_list = reward_list[:, 0]
                reward_list -= 0.5
            # 4 our
            else:
                min_reward, max_reward = reward_list.min(), reward_list.max()
                if min_reward == max_reward:
                    reward_list[:] = 0
                else:
                    reward_list = (reward_list - min_reward)/(max_reward - min_reward) - 0.5

            reward_list = reward_list.tolist()

        targets, target_idx = [], []
        if 'target_query' in dialog[turn_id]:
            for x in dialog[turn_id]['target_query']:
                x = x.lower()
                if x not in targets:
                    targets.append(x)
                    if x in query_list:
                        target_idx.append(query_list.index(x))

        return source_token_ids, source_token_type_ids, candidate_list, reward_list, query_list, target_idx


class FDialogSet(data.Dataset):
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
                        if set(candidates)&set(turn['target_query']):
                            hit_cnt += 1
                    if self.training and ('candidates' not in turn or len(lower_set(candidates))<2):
                        continue
                    instances.append((session_id, turn_id))
        print(hit_cnt/ins_cnt, hit_cnt, ins_cnt)
        return instances

    def __len__(self):
        return len(self.instance)

    def find_sublist(self, query, main_list):
        found_list = []
        j = len(query)
        for i, s in enumerate(main_list):
            if s == query[0] and query == main_list[i:i+j]:
                found_list.append((i, i+j))
        return found_list

    def get_keyword_list(self, dialog, span_size = 5):
        keyword_pool = set()
        keyword_list = []
        for utterance in dialog:
            tokens = self.tokenizer.tokenize(utterance['text'])
            for beg in range(0, len(tokens)):
                keyword = []
                cnt = 0
                if tokens[beg].startswith('##'):
                    continue
                for token in tokens[beg:]:
                    keyword.append(token)
                    if token.startswith('##'):
                        continue
                    else:
                        cnt += 1
                        if cnt < span_size + 1:
                            keyword_text = self.tokenizer.convert_tokens_to_string(keyword)
                            keyword_tokens = self.tokenizer.convert_tokens_to_ids(keyword)
                            if bool(re.search(r"[a-zA-Z]", keyword_text)) and keyword_text not in keyword_pool:
                                keyword_list.append(keyword_tokens)
                                keyword_pool.add(keyword_text)
                        else:
                            break
        return keyword_list

    def __getitem__(self, index):
        session_id, turn_id = self.instance[index]
        cls_token_id, sep_token_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        dialog = self.dataset[session_id]['dialog']

        source_token_ids = [[self.additional_special_tokens['<{}>'.format(turn['speaker'])]] + turn['tokens'] + [sep_token_id] for turn in dialog[:turn_id]]
        source_token_type_ids = [[0]*len(x) if x[0] == self.additional_special_tokens['<wizard>'] else [1]*len(x) for x in source_token_ids]
        source_token_ids = [cls_token_id] + flat_list(source_token_ids)[-self.max_length+1:]
        source_token_type_ids = [0] + flat_list(source_token_type_ids)[-self.max_length+1:]

        keyword_list = self.get_keyword_list(dialog[:turn_id])

        candidate_list = []
        query_list = []
        reward_list = []

        for keyword in keyword_list:
            found_list = self.find_sublist(keyword, source_token_ids)
            if found_list:
                query_list.append(self.tokenizer.decode(keyword))
                candidate_list.append(found_list)
                reward_list.append(0.)

        targets, target_idx = [], []
        if 'target_query' in dialog[turn_id]:
            for x in dialog[turn_id]['target_query']:
                x = x.lower()
                if x not in targets:
                    targets.append(x)
                    if x in query_list:
                        target_idx.append(query_list.index(x))

        return source_token_ids, source_token_type_ids, candidate_list, reward_list, query_list, target_idx