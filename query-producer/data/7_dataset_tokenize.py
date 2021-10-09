import os

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='0'

import json

import spacy
spacy.require_gpu()

'''
tokenize WoW dataset
'''

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')

    with open('train_wtc.json', 'r') as f:
        data = json.load(f)

    for dialog in tqdm(data):
        for turn in dialog:
            turn['tokenized_text'] = ' '.join([token.text.strip() for token in nlp(turn['text']) if len(token.text.strip())])

    with open('train_twtc.json', 'w') as f:
        json.dump(data, f)
