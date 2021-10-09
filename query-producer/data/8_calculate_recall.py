import json
import pprint
import re
import sys

from tqdm import tqdm
from transformers import AutoTokenizer
from scoring import *
import numpy as np

'''
calculate R@x for scoring functions
'''


def open_json_file(path):
    with open(path, 'r') as file:
        data=json.load(file)
    return data

def save_json_file(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file)

def main(data_path, query_path, summary_path, target_path, last_k=100, alpha=-100):
    dataset = open_json_file(data_path)
    query_dict = open_json_file(query_path)
    doc_dict = open_json_file(summary_path)
    raw_dataset = open_json_file(target_path)

    new_dataset = []
    hit_cnt, ins_cnt = 0, 0
    for example, raw_example in tqdm(zip(dataset, raw_dataset)):
        new_example = {'topic': {'text': raw_example['chosen_topic']}, 'dialog': []}
        turn_query_passages_list = []
        freq = {}
        for turn_id, (turn, raw_turn) in enumerate(zip(example[1:], raw_example['dialog'])):
            new_turn = {
                'text': turn['text'], 'speaker': turn['speaker'],
                'tokens': turn['text']
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
                    # response = [token for token in turn['tokenized_text'].split() if
                    #             token.lower() not in stop_words] + addition_tokens
                    # response = [token for token in turn['tokenized_text'].split()] + addition_tokens

                    # no coreference
                    # response = [token for token in turn['tokenized_text'].split() if token.lower() not in stop_words]

                    # bm25
                    response = [token for token in turn['tokenized_text'].split()]

                    candidates, candidate_scores = sort_query_bm25(retrieved_query_passages_list, response, alpha=alpha, freq=freq)
                    # candidates, candidate_scores = sort_query_idf(retrieved_query_passages_list, response, alpha=alpha, freq=freq)
                    new_candidates = []
                    for candidate, candidate_score in zip(candidates, candidate_scores):
                        query, passages = list(candidate.items())[0]
                        new_candidates.append(
                            {query: {'wiki_passages': [list(passage.keys())[0] for passage in passages],
                                     'tokens': query,
                                     'score': candidate_score}}
                        )
                    new_turn['candidates'] = new_candidates

                # for calculating upper bound
                if hit:
                    hit_cnt+=1
                    new_turn['target_query']=target_query_list
                ins_cnt+=1

            for token in turn['tokenized_text'].split():
                token = token.lower()
                if token in freq:
                    freq[token] += 1
                else:
                    freq[token] = 1

            turn_query_passages = []
            for mention in turn['mentions']:
                if mention['spot'] in query_dict and mention['spot'].lower() not in stop_words:
                # if mention['spot'] in query_dict and mention['lp']!=100 and mention['spot'].lower() not in stop_words:
                    spot_passages = []
                    for title in query_dict[mention['spot']]:
                        if title in doc_dict:
                            # spot_passages.append({title: flat_list(
                            #     [[token for token in sent.split() if token.lower() not in stop_words]
                            #      for sent in doc_dict[title]])})

                            spot_passages.append({title: flat_list(
                                [[token for token in sent.split()]
                                 for sent in doc_dict[title]])})
                    if spot_passages:
                        turn_query_passages.append({mention['spot']: spot_passages})
            turn_query_passages_list.append(turn_query_passages)

            new_example['dialog'].append(new_turn)
        new_dataset.append(new_example)
    print(hit_cnt/ins_cnt, hit_cnt, ins_cnt)

    result = []
    q_cnt = []
    for example in new_dataset:
        for turn in example['dialog']:
            candidates = {}
            if 'candidates' not in turn:
                continue
            for x in turn['candidates']:
                k, v = list(x.items())[0]
                k = k.lower()
                if k not in candidates:
                    candidates[k] = v['score']
            sc = sorted(candidates.keys(), key=lambda x: candidates[x], reverse=True)
            target = set([x.lower() for x in turn['target_query']]) if 'target_query' in turn else set()
            rank = 100
            for x in target:
                if x in sc:
                    rank = min(rank, sc.index(x))
            result.append(rank)
            q_cnt.append(len(candidates))
    result = np.array(result)
    print(np.mean(result < 1))
    print(np.mean(result < 3))
    print(np.mean(result < 5))
    print(np.mean(result < 100))
    print(np.mean(q_cnt))


if __name__ == '__main__':
    main('valid_twtc.json', 'wikipedia.json', 'tokenized_summary.json',
         'wizard_of_wikipedia/valid_topic_split.json')
