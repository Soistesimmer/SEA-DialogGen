import os

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='0'

import json

import spacy
spacy.require_gpu()

'''
tokenize the articles
'''


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')

    with open('summary.json', 'r') as f:
        data = json.load(f)

    new_data = {}
    for k,v in tqdm(data.items()):
        doc = nlp(v)
        new_v = []
        for sent in doc.sents:
            tokenized_sent = [token.text.strip() for token in sent if token.text.strip()!='\n']
            if len(tokenized_sent) < 128 and len(tokenized_sent) > 1:
                new_v.append(' '.join(tokenized_sent))
        new_data[k] = new_v

    with open('tokenized_summary.json', 'w') as f:
        json.dump(new_data, f)
