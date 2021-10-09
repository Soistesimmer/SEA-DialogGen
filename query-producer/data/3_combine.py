import json
import os
import pprint

'''
combine coreference resolution result with TagMe result
'word list' can directly use stopwords provided by NLTK, it is not necessary.
'''


if __name__ == '__main__':
    with open('train_wcl.json', 'r') as f:
        data = json.load(f)

    with open('train_tagme.json', 'r') as f:
        test_data = json.load(f)

    all_filter_words = set()
    for fname in os.listdir('word_list'):
        with open('word_list/{}'.format(fname), 'r') as f:
            for line in f.readlines():
                all_filter_words.add(line.strip())

    for example, test_example in zip(data, test_data):
        for turn, test_turn in zip(example['dialog'], test_example[1:]):
            if 'clusters' not in turn:
                continue
            clusters = turn['clusters']
            document = turn['document']
            if clusters:
                x = 0
                if '|||' in document:
                    x = len(document) - document[::-1].index('|||') - 1
                for cluster in clusters:
                    if cluster[-1][0] > x:
                        beg, end = cluster[0]
                        result = ' '.join(document[beg: end+1])
                        if result not in all_filter_words:
                            test_turn['mentions'].append({'spot': result, 'lp': 100})

    with open('train_wtc.json', 'w') as f:
        json.dump(test_data, f)