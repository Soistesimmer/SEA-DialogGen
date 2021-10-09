import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import pprint, json
import re

from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

'''
coreference resolution using technique from AllenNLP
'''

def format(dataset):
    new_dataset = []
    for id, example in enumerate(dataset):
        example = {
            'id': id,
            'topic': example['chosen_topic'],
            'dialog': [
                {
                    'text': add_dot(re.sub('\s+', ' ', turn['text'].strip())),
                    'speaker': turn['speaker']
                } for turn in example['dialog']
            ]
        }
        new_dataset.append(example)
    return new_dataset

def add_dot(text):
    if text[-1].isalnum():
        text += '.'
    return text

def get_doc(example, turn_id):
    return ' ||| '.join([turn['text'] for turn in example['dialog'][:turn_id+1]])

def cr(predictor, document):
    result = predictor.predict(document=document)

    document = result['document']
    clusters = result['clusters']

    return document, clusters


if __name__ == '__main__':
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
    predictor._model = predictor._model.cuda()

    file = 'wizard_of_wikipedia/train.json'

    with open(file, 'r') as f:
        dataset = json.load(f)

    data = format(dataset)

    cnt, all = 0, 0
    for example in tqdm(data):
        for turn_id, turn in enumerate(example['dialog']):
            doc = get_doc(example, turn_id)
            all += 1
            # if len(doc.split())>512:
            #     continue
            try:
                document, clusters = cr(predictor, doc)
                turn['document'] = document
                turn['clusters'] = clusters
            except:
                continue
            cnt += 1
            # if cnt%500 == 0:
            #     with open(file, 'w') as f:
            #         json.dump(data, f)
    print(cnt, all, len(data))
    
    file = 'train_wcl.json'
    with open(file, 'w') as f:
        json.dump(data, f)
