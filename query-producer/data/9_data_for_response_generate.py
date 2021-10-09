import json

'''
use this script after get the predicted queries from a query producer
'''


def main(f1, f2, f3):

    with open(f1, 'r') as f:
        data = json.load(f)

    with open(f2, 'r') as f:
        pred = json.load(f)

    print(len(pred))

    cnt = 0
    for example in data:
        for turn_id, turn in enumerate(example['dialog']):
            if turn_id > 0 and turn['speaker'] == 'wizard':
                turn['selected_query'] = pred[cnt]
                cnt += 1
    assert cnt == len(pred)

    with open(f3, 'w') as f:
        json.dump(data, f)

main('train.json', 'train_selected.json', 'train_rage.json')
main('valid.json', 'valid_selected.json', 'valid_rage.json')
main('test.json', 'test_selected.json', 'test_rage.json')