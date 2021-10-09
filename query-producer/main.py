import math
from argparse import ArgumentParser
from collections import OrderedDict

import json
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoModel, AutoTokenizer

from dataset import DialogSet, FDialogSet
from model import Model
from torch.utils.data import DataLoader
from utils import *

global_config = None

def collate_fn(batch):
    to_tensor = lambda x: [torch.tensor(y) for y in x]
    source_token_ids, source_token_type_ids, candidate_list, reward_list, queries, target_idx = zip(*batch)
    pad_token_id = global_config.pad_token_id

    candidates_mask = []
    max_len = max([len(x) for x in source_token_ids])
    for x, y in zip(source_token_ids, candidate_list):
        for candidate in y:
            candidate_mask = torch.zeros(max_len)
            for s, e in candidate:
                candidate_mask[s:e] = 1
            candidates_mask.append(candidate_mask)
    candidates_mask = pad_sequence(candidates_mask, batch_first=True, padding_value=0)

    source_token_ids = to_tensor(source_token_ids)
    source_token_type_ids = to_tensor(source_token_type_ids)

    mapping = torch.tensor(flat_list([[i]*len(x) for i, x in enumerate(reward_list)]))
    reward_list = pad_sequence(to_tensor(reward_list), batch_first=True, padding_value=-100)

    input_ids = pad_sequence(source_token_ids, batch_first=True, padding_value=pad_token_id)
    token_type_ids = pad_sequence(source_token_type_ids, batch_first=True, padding_value=0)
    attention_mask = (input_ids != pad_token_id).long()

    batch = {
        'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
        'candidate_mask': candidates_mask, 'mapping': mapping, 'rewards': reward_list
    }
    return batch, (queries,target_idx)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--train_data", type=str)
    argparser.add_argument("--valid_data", type=str)
    argparser.add_argument("--test_data", type=str)
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--output_path", type=str)
    argparser.add_argument('--refresh_step', type=int, default=1)
    argparser.add_argument('--epoch', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--gradient_accumulate', type=int, default=4)
    argparser.add_argument('--valid_step', type=int, default=200)
    argparser.add_argument('--lr', type=float, default=1e-5)
    argparser.add_argument('--early_stop', type=int, default=3)
    argparser.add_argument('--max_cx_len', type=int, default=512)
    argparser.add_argument('--max_pos_emb_size', type=int, default=512)
    argparser.add_argument('--max_length', type=int, default=50)
    argparser.add_argument('--min_length', type=int, default=0)
    argparser.add_argument('--num_beams', type=int, default=4)
    argparser.add_argument('--training', action='store_true')
    argparser.add_argument('--ppl', action='store_true')
    argparser.add_argument('--beg_rl', type=int, default=-1)
    argparser.add_argument('--continue_training', action='store_true')
    args = argparser.parse_args()

    # bb = 'google/electra-small-discriminator'
    bb = 'google/electra-base-discriminator'
    tokenizer = AutoTokenizer.from_pretrained(bb)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<wizard>', '<apprentice>']})
    args.additional_special_tokens = {k: v for k, v in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)}
    args.vocab_size = len(tokenizer.get_vocab())

    bb_model = AutoModel.from_pretrained(bb)
    bb_model.resize_token_embeddings(args.vocab_size)

    args.pad_token_id = tokenizer.pad_token_id
    global_config = args

    model = Model(args, bb_model)

    if args.training:
        train_set = DialogSet(args, args.train_data, tokenizer)
        valid_set = DialogSet(args, args.valid_data, tokenizer)
        train_loader = DataLoader(train_set, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
        valid_loader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)

        if args.continue_training:
            model.load_state_dict(torch.load(args.model_path))

        optimizer = AdamW(model.parameters(), lr=args.lr)
        model = model.cuda()

        best_reward, step, temper = -1., 0, 0
        model = model.train()
        for epoch in range(args.epoch):
            train_watcher = tqdm(enumerate(train_loader))
            train_watcher.set_description('Epoch {}, Train'.format(epoch))
            train_loss = 0.
            use_rl = epoch > args.beg_rl
            for i, batch in train_watcher:
                step += 1
                batch = batch_to_gpu(batch[0])
                loss = model(batch, use_rl)[1]/args.gradient_accumulate
                train_loss += loss.item()
                loss.backward()

                if step % args.gradient_accumulate == 0:
                    backward_step = step//args.gradient_accumulate
                    optimizer.step()
                    optimizer.zero_grad()

                    if backward_step % args.refresh_step==0:
                        train_watcher.set_postfix(OrderedDict(step=backward_step, loss='{:.3f}'.format(train_loss/args.refresh_step)), refresh=True)
                        train_loss = 0.

                    if backward_step % args.valid_step == 0:
                        model = model.eval()
                        valid_reward = 0.
                        for j, batch in enumerate(valid_loader):
                            batch = batch_to_gpu(batch[0])
                            with torch.no_grad():
                                pred = model(batch)[0]
                            for x, y in zip(pred, batch['rewards']):
                                valid_reward += y[torch.argmax(x)]
                        valid_reward/= j*args.batch_size
                        print('\nEvaluating, Step {}, Mean Reward {:.4f}'.format(backward_step, valid_reward))
                        if best_reward < valid_reward:
                            best_reward, temper = valid_reward, 0
                            torch.save(model.state_dict(), args.model_path)
                        else:
                            temper += 1
                            if temper > args.early_stop:
                                break
                        model = model.train()
            if temper > args.early_stop:
                break

    else:
        ###
        # no space pruning
        # valid_set = FDialogSet(args, args.test_data, tokenizer)
        # valid_loader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)
        #
        # model.load_state_dict(torch.load(args.model_path))
        # model = model.cuda()
        # model = model.eval()
        #
        # valid_watcher = tqdm(enumerate(valid_loader))
        # selected = []
        # log = []
        # for i, (batch, target) in valid_watcher:
        #     batch = batch_to_gpu(batch)
        #     pred, loss = model(batch)
        #     for x, y, z in zip(pred, target[1], target[0]):
        #         _log = 100
        #         x = x[:len(z)]
        #         for _y in y:
        #             _log = min(torch.sum(x[_y] < x).item(), _log)
        #         log.append(_log)
        #         if z:
        #             selected.append([z[j] for j in x.argsort(descending=True)[:5]])
        #         else:
        #             selected.append([])
        # log = np.array(log)
        # for i in range(5):
        #     print('R@{}'.format(i + 1), np.mean(log < i + 1))
        # with open(args.output_path, 'w') as f:
        #     json.dump(selected, f)
        # exit()
        ###

        valid_set = DialogSet(args, args.test_data, tokenizer)
        valid_loader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)

        model.load_state_dict(torch.load(args.model_path))
        model = model.cuda()
        model = model.eval()

        valid_watcher = tqdm(enumerate(valid_loader))
        selected = []
        log = []
        for i, (batch, target) in valid_watcher:
            batch = batch_to_gpu(batch)
            pred, loss = model(batch)
            for x, y, z in zip(pred, target[1], target[0]):
                _log = 100
                x = x[:len(z)]
                for _y in y:
                    _log = min(torch.sum(x[_y]<x).item(), _log)
                log.append(_log)
                if z:
                    selected.append([z[j] for j in x.argsort(descending=True)[:5]])
                else:
                    selected.append([])
        log = np.array(log)
        for i in range(5):
            print('R@{}'.format(i+1), np.mean(log<i+1))
        with open(args.output_path, 'w') as f:
            json.dump(selected, f)
        exit()

        valid_watcher = tqdm(enumerate(valid_loader))
        selected = []
        r, t = 0, 0
        for i, (batch, target) in valid_watcher:
            batch = batch_to_gpu(batch)
            pred, loss = model(batch)
            for x, y, z in zip(pred, target[1], target[0]):
                t += 1
                if x.argmax() in y:
                    r += 1
                if z:
                    selected.append(z[x.argmax()])
                else:
                    selected.append([])
        print(r, t, r/t)
        with open(args.output_path, 'w') as f:
            json.dump(selected, f)