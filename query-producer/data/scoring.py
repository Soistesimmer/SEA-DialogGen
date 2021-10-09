from rank_bm25 import BM25Plus
from nltk.corpus import stopwords
import random, json
import numpy as np

'''
contain some scoring functions
'''

stop_words = set(stopwords.words('english'))
other_special_tokens = {',','.','!','\'','?','(',')'}
stop_words.update(other_special_tokens)

# if TF-IDF, please prepare idf from training dataset first
#with open('data/idf.json', 'r') as f:
#   idf = json.load(f)

def sort_query(query_passages_list, response, alpha=-100):
    selected_queries, selected_query_scores = [], []
    for query_passages in query_passages_list:
        query, passages = list(query_passages.items())[0]
        query_score = sort_passage(passages, response)
        if query_score > alpha:
            selected_queries.append(query_passages)
            selected_query_scores.append(query_score)
    sorted_results = sorted(zip(selected_queries, selected_query_scores), key=lambda x: x[1], reverse=True)
    sorted_query_passages_list, sorted_query_scores = zip(*sorted_results) if sorted_results else ([],[])
    return sorted_query_passages_list, sorted_query_scores

def sort_query_bm25(queries, response, alpha=-100, **kwarg):
    _queries, passages = zip(*[list(query.items())[0] for query in queries])
    q_i_dict = {}
    _passages = []
    cnt = 0
    for i, passage in enumerate(passages):
        for _passage in passage:
            _passages.append(list(_passage.values())[0])
            q_i_dict[cnt] = i
            cnt += 1
    bm25 = BM25Plus(_passages)
    scores = bm25.get_scores(response)
    query_scores = [-100]*len(_queries)
    for i, score in enumerate(scores):
        if query_scores[q_i_dict[i]]<score:
            query_scores[q_i_dict[i]] = score
    return zip(*sorted(zip(queries, query_scores), key=lambda x:x[1], reverse=True))

def sort_passage(passages, response):
    passage_scores = []
    for passage in passages:
        score = calculate_score(passage, response)
        passage_scores.append(score)
    return max(passage_scores)


def sort_query_random(queries, response, alpha=-100, **kwargs):
    random.shuffle(queries)
    return queries, [0]*len(queries)


def idf_fn(x, freq):
    score = []
    for y in x.split():
        score.append((idf[y] if y in idf else idf['<unk>'])*(freq[y] if freq and y in freq else 1))
    return np.mean(score)

def sort_query_idf(queries, response, alpha=-100, freq=None):
    query_scores = [idf_fn(list(query.keys())[0].lower(), freq) for query in queries]
    return zip(*sorted(zip(queries, query_scores), key=lambda x: x[1], reverse=True))


def sort_query_kw(queries, response, alpha=-100, freq=None):
    response = ' '.join(response).lower()
    # query_scores = zip([1 if list(query.keys())[0].lower() in response else 0 for query in queries],
    #                    [idf_fn(list(query.keys())[0].lower(), freq) for query in queries])
    query_scores = [(1 if list(query.keys())[0].lower() in response else 0, idf_fn(list(query.keys())[0].lower(), freq)) for query in queries]
    # query_scores = [1 if list(query.keys())[0].lower() in response else 0 for query in queries]
    return zip(*sorted(zip(queries, query_scores), key=lambda x: x[1], reverse=True))


def sort_query_arr(queries, response, alpha=-100, freq=None):
    bm25 = BM25Plus([[x.lower() for x in response]])
    scores = []
    for query in queries:
        score = bm25.get_scores(list(query.keys())[0].lower().split())
        scores.append(score)
    return zip(*sorted(zip(queries, scores), key=lambda x: x[1], reverse=True))



def calculate_score(passage, response):
    x = list(passage.values())[0]
    if x:
        assert isinstance(x[0], str)
        bm25 = BM25Plus([x])
        scores = bm25.get_scores(response)
        return max(scores)
    else:
        return 0

def flat_list(input):
    output = []
    for x in input:
        output += x
    return output