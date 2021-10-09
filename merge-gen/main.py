import math
from argparse import ArgumentParser
from collections import OrderedDict

import json
import torch
from torch.nn import DataParallel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AdamW, BartModel
from transformers.models.bart import BartForConditionalGeneration, BartTokenizer
from dataset_fid_v2 import DialogSet

from torch.utils.data import DataLoader
from utils import *
from model_fid_v2 import FiD

global_config = None
torch.autograd.set_detect_anomaly(True)

def collate_fn(batch):
    input_ids_list, response_token_ids = zip(*batch)
    pad_token_id = global_config.pad_token_id
    pad_label_id = global_config.pad_label_id
    max_kn_size = max([len(x) for x in input_ids_list])
    max_kn_input_len = max([len(y) for x in input_ids_list for y in x])
    input_ids = torch.ones(len(input_ids_list), max_kn_size, max_kn_input_len).long()*pad_token_id
    for i, x in enumerate(input_ids_list):
        for j, y in enumerate(x):
            input_ids[i, j, :len(y)] = torch.tensor(y)
    attention_mask = (input_ids!=pad_token_id).long()
    crx_attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)
    decoder_input_ids = pad_sequence([torch.tensor(example)[:-1] for example in response_token_ids], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([torch.tensor(example)[1:] for example in response_token_ids], batch_first=True, padding_value=pad_label_id)
    batch = {
        'input_ids': input_ids, 'attention_mask': attention_mask,
        'crx_attention_mask': crx_attention_mask,
        'decoder_input_ids': decoder_input_ids, 'labels': labels,
    }
    return batch


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--train_data", type=str)
    argparser.add_argument("--valid_data", type=str)
    argparser.add_argument("--test_data", type=str)
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--output_path", type=str)
    argparser.add_argument("--type", type=str, default='bm25_top10')
    argparser.add_argument('--refresh_step', type=int, default=1)
    argparser.add_argument('--epoch', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--gradient_accumulate', type=int, default=1)
    argparser.add_argument('--valid_step', type=int, default=500)
    argparser.add_argument('--lr', type=float, default=1e-5)
    argparser.add_argument('--early_stop', type=int, default=5)
    argparser.add_argument('--max_cx_len', type=int, default=256)
    argparser.add_argument('--max_kn_len', type=int, default=128)
    argparser.add_argument('--max_length', type=int, default=64)
    argparser.add_argument('--min_length', type=int, default=10)
    argparser.add_argument('--num_beams', type=int, default=4)
    argparser.add_argument('--topk', type=int, default=100)
    argparser.add_argument('--dp', action='store_true')
    argparser.add_argument('--training', action='store_true')
    argparser.add_argument('--ppl', action='store_true')
    argparser.add_argument('--continue_training', action='store_true')
    args = argparser.parse_args()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<wizard>', '<apprentice>', '<dialog>', '<knowledge>']})
    args.additional_special_tokens = {k: v for k, v in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)}
    args.vocab_size = len(tokenizer.get_vocab())

    model = FiD.from_pretrained('facebook/bart-base')
    model.resize_token_embeddings(args.vocab_size)

    model.model.equip()

    args.pad_token_id = tokenizer.pad_token_id
    args.pad_label_id = -100
    global_config = args

    if args.training:
        train_set = DialogSet(args.train_data, tokenizer, args)
        valid_set = DialogSet(args.valid_data, tokenizer, args)
        train_loader = DataLoader(train_set, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
        valid_loader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)

        if args.continue_training:
            model.load_state_dict(torch.load(args.model_path))

        optimizer = AdamW(model.parameters(), lr=args.lr)
        model = model.cuda()
        if args.dp:
            model = DataParallel(model)

        best_loss, step, temper = 1e6, 0, 0
        model = model.train()
        for epoch in range(args.epoch):
            train_watcher = tqdm(enumerate(train_loader))
            train_watcher.set_description('Epoch {}, Train'.format(epoch))
            train_loss = 0.
            for i, batch in train_watcher:
                step += 1
                batch = batch_to_gpu(batch)
                loss = model(**batch).loss.mean()/args.gradient_accumulate
                train_loss += loss.item()
                loss.backward()

                if step % args.gradient_accumulate == 0:
                    backward_step = step//args.gradient_accumulate
                    optimizer.step()
                    optimizer.zero_grad()

                    if backward_step % args.refresh_step==0:
                        train_watcher.set_postfix(OrderedDict(step=backward_step, loss='{:.3f}'.format(train_loss/args.refresh_step),
                                                              ), refresh=True)
                        train_loss = 0.

                    if backward_step % args.valid_step == 0:
                        model = model.eval()
                        valid_watcher = tqdm(enumerate(valid_loader))
                        valid_watcher.set_description('Step {}, Valid'.format(backward_step))
                        valid_loss = 0.
                        for j, batch in valid_watcher:
                            batch = batch_to_gpu(batch)
                            with torch.no_grad():
                                loss = model(**batch).loss.mean()
                            valid_loss += loss.item()
                            if (j+1)%args.refresh_step==0:
                                valid_watcher.set_postfix(loss='{:.3f}'.format(valid_loss/(j+1)), refresh=True)
                        valid_loss/=j
                        if best_loss > valid_loss:
                            best_loss, temper = valid_loss, 0
                            torch.save(model.module.state_dict() if args.dp else model.state_dict(), args.model_path)
                        else:
                            temper += 1
                            if temper > args.early_stop:
                                break
                        model = model.train()
            if temper > args.early_stop:
                break

    else:
        valid_set = DialogSet(args.test_data, tokenizer, args)
        valid_loader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)

        try:
            model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
        except:
            state_dict = OrderedDict()
            checkpoint_state_dict = torch.load(args.model_path, map_location='cuda:0')
            for k, v in checkpoint_state_dict.items():
                k = k.replace('module.', '')
                state_dict[k] = v
            model.load_state_dict(state_dict)
        model = model.cuda()
        model = model.eval()

        if args.ppl:
            valid_watcher = tqdm(enumerate(valid_loader))
            valid_loss, valid_cnt = 0., 0.
            for i, batch in valid_watcher:
                batch = batch_to_gpu(batch)
                with torch.no_grad():
                    loss = model(**batch).loss
                i_cnt = torch.sum(batch['labels'] > 0)
                valid_loss += loss.item() * i_cnt
                valid_cnt += i_cnt
            valid_ppl = math.exp(valid_loss / valid_cnt)
            print('Test PPL: {:.4f}'.format(valid_ppl))
        else:
            predictions, references = [], []
            valid_watcher = tqdm(enumerate(valid_loader))
            for i, batch in valid_watcher:
                batch = batch_to_gpu(batch)
                decoder_input_ids = batch['decoder_input_ids']
                del batch['decoder_input_ids']
                del batch['labels']
                with torch.no_grad():
                    output_ids = model.generate(**batch, num_beams=args.num_beams, max_length=args.max_length,
                                                min_length=args.min_length, early_stopping=True)
                predictions += tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                references += tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
            unigram_f1, f1_scores = evaluate_uni_f1(predictions, references)
            print('Test Unigram F1: {:.4f}'.format(unigram_f1))
            with open(args.output_path, 'w') as file:
                json.dump([{'pred': item[0], 'ref': item[1], 'f1': f1_scores[i].value()}
                           for i, item in enumerate(zip(predictions, references))], file)