import json
import pprint
import re
import sys

from tqdm import tqdm
from transformers import AutoTokenizer
from data.scoring import *

'''
prepare data for query producers
'''

def open_json_file(path):
    with open(path, 'r') as file:
        data=json.load(file)
    return data

def save_json_file(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file)

def main(data_path, query_path, summary_path, target_path, output_path, tokenizer, last_k=100, alpha=-100):
    dataset = open_json_file(data_path)
    query_dict = open_json_file(query_path)
    doc_dict = open_json_file(summary_path)
    raw_dataset = open_json_file(target_path)

    new_dataset = []
    hit_cnt, ins_cnt = 0, 0
    for example, raw_example in tqdm(zip(dataset, raw_dataset)):
        new_example = {'topic': {'text': raw_example['chosen_topic']}, 'dialog': []}
        turn_query_passages_list = []
        for turn_id, (turn, raw_turn) in enumerate(zip(example[1:], raw_example['dialog'])):
            new_turn = {
                'text': turn['text'], 'speaker': turn['speaker'],
                'tokens': tokenizer.encode(turn['text'], add_special_tokens=False)
            }
            if turn['speaker'] == 'wizard' and turn_id > 0:
                target_passage_title, target_query_list, hit = None, [], False
                if 'checked_passage' in raw_turn and raw_turn['checked_passage'] and isinstance(raw_turn['checked_passage'], dict):
                    target_passage_title = list(raw_turn['checked_passage'].values())[0]
                    if target_passage_title != 'no_passages_used':
                        new_turn['target_passage'] = target_passage_title
                    else:
                        target_passage_title = None
                retrieved_query_set = set()
                retrieved_query_passages_list = []
                for query_passages_list in flat_list(turn_query_passages_list[-last_k:]):
                    query, passages = list(query_passages_list.items())[0]
                    if query not in retrieved_query_set:
                        retrieved_query_set.add(query)
                        retrieved_query_passages_list.append(query_passages_list)
                        for passage in passages:
                            if target_passage_title == list(passage.keys())[0]:
                                hit=True
                                target_query_list.append(query)
                                break
                if retrieved_query_passages_list:
                    # coreference and stopword
                    addition_tokens = flat_list([mention['spot'].split() for mention in turn['mentions']
                                                 if mention['lp'] == 100 and mention['spot'].lower() not in stop_words])
                    response = [token for token in turn['tokenized_text'].split() if
                                token.lower() not in stop_words] + addition_tokens

                    # no coreference
                    # response = [token for token in turn['tokenized_text'].split() if token.lower() not in stop_words]

                    # bm25
                    # response = [token for token in turn['tokenized_text'].split()]

                    candidates, candidate_scores = sort_query_bm25(retrieved_query_passages_list, response, alpha=alpha)
                    new_candidates = []
                    for candidate, candidate_score in zip(candidates, candidate_scores):
                        query, passages = list(candidate.items())[0]
                        new_candidates.append(
                            {query: {'wiki_passages': [list(passage.keys())[0] for passage in passages],
                                     'tokens': tokenizer.encode(query, add_special_tokens=False),
                                     'score': candidate_score}}
                        )
                    new_turn['candidates'] = new_candidates

                # for calculating upper bound
                if hit:
                    hit_cnt+=1
                    new_turn['target_query']=target_query_list
                ins_cnt+=1

            turn_query_passages = []
            for mention in turn['mentions']:
                if mention['spot'] in query_dict and mention['spot'].lower() not in stop_words:
                # if mention['spot'] in query_dict and mention['lp']!=100:
                    spot_passages = []
                    for title in query_dict[mention['spot']]:
                        if title in doc_dict:
                            spot_passages.append({title: flat_list(
                                [[token for token in sent.split() if token.lower() not in stop_words]
                                 for sent in doc_dict[title]])})

                            # spot_passages.append({title: flat_list(
                            #     [[token for token in sent.split()]
                            #      for sent in doc_dict[title]])})
                    if spot_passages:
                        turn_query_passages.append({mention['spot']: spot_passages})
            turn_query_passages_list.append(turn_query_passages)

            new_example['dialog'].append(new_turn)
        new_dataset.append(new_example)
    print(hit_cnt/ins_cnt, hit_cnt, ins_cnt)
    save_json_file(new_dataset, output_path)


if __name__ == '__main__':
    # generation-based
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    main('data/test_twtc.json', 'data/wikipedia.json', 'data/tokenized_summary.json',
         '../wizard_of_wikipedia/test_topic_split.json', 'test_gen.json', tokenizer)
    main('data/valid_twtc.json', 'data/wikipedia.json', 'data/tokenized_summary.json',
         '../wizard_of_wikipedia/valid_topic_split.json', 'valid_gen.json', tokenizer)
    main('data/train_twtc.json', 'data/wikipedia.json', 'data/tokenized_summary.json',
         '../wizard_of_wikipedia/train.json', 'train_gen.json', tokenizer)
    
    # extraction-based
    # tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    # main('data/test_twtc.json', 'data/wikipedia.json', 'data/tokenized_summary.json',
    #      '../wizard_of_wikipedia/test_topic_split.json', 'test.json', tokenizer)
    # main('data/valid_twtc.json', 'data/wikipedia.json', 'data/tokenized_summary.json',
    #      '../wizard_of_wikipedia/valid_topic_split.json', 'valid.json', tokenizer)
    # main('data/train_twtc.json', 'data/wikipedia.json', 'data/tokenized_summary.json',
    #      '../wizard_of_wikipedia/train.json', 'train.json', tokenizer)