import json
import pprint
import re
import sys

from tqdm import tqdm
from transformers import AutoTokenizer
from data.scoring import *

def open_json_file(path):
    with open(path, 'r') as file:
        data=json.load(file)
    return data

def save_json_file(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file)


def main(data_path, tkn_data_path, summary_path, output_path, tokenizer, topk=10, topq=1):
    dataset = open_json_file(data_path)
    tkn_dataset = open_json_file(tkn_data_path)
    doc_dict = open_json_file(summary_path)

    tkn_fn = lambda x: [{'text':y, 'tokens': tokenizer.encode(y, add_special_tokens=False)} for y in x]

    for example, tkn_example in tqdm(zip(dataset, tkn_dataset)):
        example['topic']['tokens'] = tokenizer.encode(example['topic']['text'], add_special_tokens=False)
        for turn_id, (turn, tkn_turn) in enumerate(zip(example['dialog'], tkn_example[1:])):
            turn['tokens'] = tokenizer.encode(turn['text'], add_special_tokens=False)
            if turn['speaker'] == 'wizard' and turn_id > 0 and turn['selected_query']:
                raw_titles, selected_titles = set(), set()
                for candidate in turn['candidates']:
                    k, v = list(candidate.items())[0]
                    if k.lower() in turn['selected_query'][:topq]:
                        selected_titles.update(set(v['wiki_passages']))
                    raw_titles.update(set(v['wiki_passages']))
                query_1 = ' '.join([x['tokenized_text'] for x in tkn_example[1:][:turn_id][-2:]] +
                                   [mention['spot'] for _turn in tkn_example[1:][:turn_id][-2:]
                                    for mention in _turn['mentions'] if mention['lp'] == 100])  # coreference
                raw_passages = [' '.join([sent for sent in doc_dict[title][:5] if len(sent.split()) < 100]) for title in
                                raw_titles]
                selected_passages = [' '.join([sent for sent in doc_dict[title][:5] if len(sent.split()) < 100]) for
                                     title in selected_titles]
                if raw_passages:
                    turn['bm25'] = tkn_fn(get_top_n(query_1, raw_passages, topk))
                if selected_passages:
                    turn['pred_bm25'] = tkn_fn(get_top_n(query_1, selected_passages, topk))

    save_json_file(dataset, output_path)

# doc multi. query
if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    main('../test_rage.json', 'data/test_twtc.json', 'data/tokenized_summary.json', '../test_fid_20_1.json', tokenizer, 20, 1)
    main('../valid_rage.json', 'data/valid_twtc.json', 'data/tokenized_summary.json', '../valid_fid_20_1.json', tokenizer, 20, 1)
    main('../train_rage.json', 'data/train_twtc.json', 'data/tokenized_summary.json', '../train_fid_20_1.json', tokenizer, 20, 1)
