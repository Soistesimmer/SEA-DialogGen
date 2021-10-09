import json
import os
import threading

import wikipedia

'''
get wikipedia page titles for prepared all queries candidates in dataset
'''

class dataset:
    def __init__(self, kps):
        self.kps = kps
        if os.path.exists('wikipedia.json'):
            with open('wikipedia.json', 'r') as f:
                self.wiki_base = json.load(f)
        else:
            self.wiki_base = {}
        self.cnt, self.total = 0, sum([1 for kp in self.kps if kp not in self.wiki_base])
        self.data_pool = self.yield_example()

    def yield_example(self):
        for kp in self.kps:
            if kp not in self.wiki_base:
                yield kp

    def save(self):
        with open('wikipedia.json', 'w') as f:
            json.dump(self.wiki_base, f)

def thread_fn(name, lock_in, lock_out):
    global data
    while True:
        with lock_in:
            try:
                kp = next(data.data_pool)
            except StopIteration:
                break
            data.cnt += 1
            if data.cnt % 10 == 0:
                print(data.cnt, data.total)
        try:
            results = wikipedia.search(kp, results=5)
            with lock_out:
                data.wiki_base[kp] = results
                if data.cnt % 50 == 0:
                    data.save()
        except:
            continue

kps = set()

with open('train_wtc.json', 'r') as f:
    train_data = json.load(f)
with open('valid_wtc.json', 'r') as f:
    valid_data = json.load(f)
with open('test_wtc.json', 'r') as f:
    test_data = json.load(f)

for dialog in (train_data+valid_data+test_data):
    for turn in dialog[1:]:
        if 'mentions' in turn:
            for kp in turn['mentions']:
                if kp['lp'] > 0.01:
                    kps.add(kp['spot'])

# with open('test_wtc.json', 'r') as f:
#     test_data = json.load(f)
#
# for dialog in test_data:
#     for turn in dialog[1:]:
#         if 'mentions' in turn:
#             for kp in turn['mentions']:
#                 if kp['lp']>0.01:
#                     kps.add(kp['spot'])

print(len(kps))
data = dataset(kps)

thread_num = 10
threads = []
lock_in = threading.Lock()
lock_out = threading.Lock()
for i in range(thread_num):
    new_thread = threading.Thread(target=thread_fn, args=('Thread-{}'.format(i), lock_in, lock_out))
    new_thread.start()
    threads.append(new_thread)
for thread in threads:
    thread.join()

data.save()

