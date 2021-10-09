import json
import os
import threading

import wikipedia

'''
retrieve wikipedia page content (summary or the first paragraph)
'''


class dataset:
    def __init__(self, kps):
        self.kps = kps
        if os.path.exists('summary.json'):
            with open('summary.json', 'r') as f:
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
        with open('summary.json', 'w') as f:
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
                print(data.cnt, data.total, len(data.wiki_base))
        try:
            results = wikipedia.summary(kp, auto_suggest=False)
            with lock_out:
                data.wiki_base[kp] = results
                if data.cnt % 50 == 0:
                    data.save()
        except:
            continue

kps = set()

with open('wikipedia.json', 'r') as f:
    data = json.load(f)

for k,v in data.items():
    for x in v:
        kps.add(x)

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

