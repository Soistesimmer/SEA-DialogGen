import json
import time

import requests
import threading
from requests import HTTPError

from tqdm import tqdm

'''
get query candidates using TagMe, remember to apply for your own TOKEN from website
'''

def open_json_file(path):
    with open(path, 'r') as file:
        data=json.load(file)
    return data

def save_json_file(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file)

def get_mentions(text):
    MY_GCUBE_TOKEN = 'YOUR TOKEN'
    tagme_url = 'https://tagme.d4science.org/tagme/spot'
    payload = [("gcube-token", MY_GCUBE_TOKEN),
               ("text", text),
               ("lang", 'en')]

    response = requests.get(tagme_url, params=payload)
    response.raise_for_status()
    response = sorted(response.json()['spots'], key=lambda x:x['lp'],reverse=True)
    return response

def get_data_multi_thread(dataset, num=15):
    def data_generator():
        for i, example in tqdm(enumerate(dataset)):
            for j, turn in enumerate(example):
                if turn['mentions'] is None:
                    yield {
                        'session': i,
                        'turn': j,
                        'text': turn['text'],
                    }
    data_pool = data_generator()
    def thread_fn(name, lock):
        while True:
            with lock:
                try:
                    example = next(data_pool)
                except StopIteration:
                    break
            mentions = None
            while True:
                down_time = 0
                try:
                    mentions = get_mentions(example['text'])
                    break
                except HTTPError as e:
                    down_time += 1
                    if down_time > 10:
                        print("{}: frequently down (>{}), return None instead".format(name, down_time))
                        break
                    pause = int(e.headers["Retry-After"])+1 if e.headers["Retry-After"] else 2**down_time
                    print("{}: {}, restart {}s later".format(name, e, pause))
                    time.sleep(pause)
            dataset[example['session']][example['turn']]['mentions'] = mentions

    threads = []
    lock = threading.Lock()
    for i in range(num):
        new_thread = threading.Thread(target=thread_fn, args=('Thread-{}'.format(i), lock))
        new_thread.start()
        threads.append(new_thread)
    for thread in threads:
        thread.join()
    return dataset

if __name__ == '__main__':

    dataset = open_json_file("wizard_of_wikipedia/valid_topic_split.json")

    new_dataset = []
    for session_id, instance in enumerate(dataset):
        new_instance = [{'speaker': 'topic', 'mentions': [instance['chosen_topic']],
                         'text': instance['chosen_topic']}]

        for turn in instance['dialog']:
            new_instance.append({'speaker': turn['speaker'][2:].lower(), 'text': turn['text'],
                                 'mentions': None})

        new_dataset.append(new_instance)

    output = get_data_multi_thread(new_dataset)
    save_json_file(output, 'valid_tagme.json')