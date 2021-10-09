import math
from argparse import ArgumentParser
from collections import OrderedDict

import json
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AdamW
from transformers.models.bart import BartForConditionalGeneration, BartTokenizer
from dataset_gen import DialogSet
from model_gen import Generator
from torch.utils.data import DataLoader
from utils import *

global_config = None

def collate_fn(batch):
    to_tensor = lambda x: [torch.tensor(y) for y in x]
    source_token_ids, target_token_ids_list, reward_list, target_indices = zip(*batch)
    pad_token_id = global_config.pad_token_id
    pad_label_id = global_config.pad_label_id
    source_token_ids = to_tensor(source_token_ids)
    target_token_ids = to_tensor(flat_list(target_token_ids_list))
    mapping = torch.tensor(flat_list([[i]*len(x) for i, x in enumerate(reward_list)]))
    reward_list = torch.tensor(flat_list(reward_list))
    input_ids = pad_sequence(source_token_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = (input_ids != pad_token_id).long()
    decoder_input_ids = pad_sequence([example[:-1] for example in target_token_ids], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([example[1:] for example in target_token_ids], batch_first=True,padding_value=pad_label_id)
    target_mask = len_list_to_mask([len(x) for x in target_token_ids_list])
    batch = {
        'input_ids': input_ids, 'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids, 'labels': labels,
        'mapping': mapping, 'rewards': reward_list,
        'target_mask': target_mask
    }
    return batch, target_indices


def collate_fn_4t(batch):
    to_tensor = lambda x: [torch.tensor(y) for y in x]
    source_token_ids, targets, target_indices = zip(*batch)
    pad_token_id = global_config.pad_token_id
    source_token_ids = to_tensor(source_token_ids)
    input_ids = pad_sequence(source_token_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = (input_ids != pad_token_id).long()
    batch = {
        'input_ids': input_ids, 'attention_mask': attention_mask,
        'targets': targets, "target_indices": target_indices
    }
    return batch


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--train_data", type=str)
    argparser.add_argument("--valid_data", type=str)
    argparser.add_argument("--test_data", type=str)
    argparser.add_argument("--checkpoint", type=str)
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--output_path", type=str)
    argparser.add_argument('--refresh_step', type=int, default=1)
    argparser.add_argument('--epoch', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=16)
    argparser.add_argument('--gradient_accumulate', type=int, default=4)
    argparser.add_argument('--valid_step', type=int, default=200)
    argparser.add_argument('--lr', type=float, default=1e-5)
    argparser.add_argument('--early_stop', type=int, default=4)
    argparser.add_argument('--max_cx_len', type=int, default=512)
    argparser.add_argument('--max_pos_emb_size', type=int, default=512)
    argparser.add_argument('--max_length', type=int, default=50)
    argparser.add_argument('--min_length', type=int, default=0)
    argparser.add_argument('--num_beams', type=int, default=10)
    argparser.add_argument('--num_return_sequences', type=int, default=5)
    argparser.add_argument('--beg_rl', type=int, default=0)
    argparser.add_argument('--beg_val', type=int, default=0)
    argparser.add_argument('--training', action='store_true')
    argparser.add_argument('--ppl', action='store_true')
    argparser.add_argument('--given', action='store_true')
    argparser.add_argument('--continue_training', action='store_true')
    args = argparser.parse_args()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<wizard>', '<apprentice>', '<no_passage_used>', '<emt>']})
    args.additional_special_tokens = {k: v for k, v in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)}
    args.vocab_size = len(tokenizer.get_vocab())

    model = Generator.from_pretrained('facebook/bart-base')
    model.resize_token_embeddings(args.vocab_size)

    args.pad_token_id = tokenizer.pad_token_id
    args.pad_label_id = -100
    global_config = args

    if args.training:
        train_set = DialogSet(args, args.train_data, tokenizer)
        valid_set = DialogSet(args, args.valid_data, tokenizer)
        train_loader = DataLoader(train_set, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
        # valid_loader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)
        valid_set.training = False
        valid_loader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn_4t, batch_size=args.batch_size)

        if args.continue_training:
            model.load_state_dict(torch.load(args.checkpoint))

        optimizer = AdamW(model.parameters(), lr=args.lr)
        model = model.cuda()

        best_reward, step, temper = -1, 0, 0
        gradient_accumulate = args.gradient_accumulate
        valid_step = args.valid_step
        model = model.train()
        for epoch in range(args.epoch):
            train_watcher = tqdm(enumerate(train_loader))
            train_watcher.set_description('Epoch {}, Train'.format(epoch))
            train_loss = 0.
            use_rl = epoch > args.beg_rl
            for i, batch in train_watcher:
                step += 1
                batch = batch_to_gpu(batch[0])
                loss = model(**batch, use_rl=use_rl, ftft=False).loss#+model(**batch, use_rl=use_rl, ftft=True).loss
                loss /= args.gradient_accumulate
                if isinstance(loss, torch.Tensor):
                    train_loss += loss.item()
                    loss.backward()

                if step % gradient_accumulate == 0:
                    backward_step = step//gradient_accumulate
                    optimizer.step()
                    optimizer.zero_grad()

                    if backward_step % args.refresh_step==0:
                        train_watcher.set_postfix(OrderedDict(step=backward_step, loss='{:.3f}'.format(train_loss/args.refresh_step)), refresh=True)
                        train_loss = 0.
                    if backward_step % valid_step == 0 and epoch > args.beg_val:
                        print('\nvalidating..')
                        model = model.eval()
                        log = []
                        for i, batch in enumerate(valid_loader):
                            batch = batch_to_gpu(batch)
                            output_ids = model.generate(batch['input_ids'], attention_mask=batch['attention_mask'],
                                                        num_beams=args.num_beams, max_length=args.max_length,
                                                        min_length=args.min_length, early_stopping=True,
                                                        num_return_sequences=args.num_return_sequences)
                            b_pred = []
                            j = 0
                            for x in tokenizer.batch_decode(output_ids, skip_special_tokens=True):
                                b_pred.append(x.strip().lower())
                                if len(b_pred) == args.num_return_sequences:
                                    b_pred = filter_repeat_kp(b_pred)
                                    rank = [100]
                                    for y in batch['targets'][j]:
                                        y = y.lower()
                                        if y in b_pred:
                                            rank.append(b_pred.index(y))
                                    log.append(min(rank))
                                    b_pred = []
                                    j += 1
                            # predictions += tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                        log = np.array(log)
                        for i in range(5):
                            print('R@{}'.format(i + 1), np.mean(log < i + 1))
                        r1 = np.mean(log < 1)

                        # valid_watcher = tqdm(enumerate(valid_loader))
                        # valid_watcher.set_description('Step {}, Valid'.format(backward_step))
                        # valid_reward, valid_loss = 0., 0.
                        # for j, batch in valid_watcher:
                        #     batch = batch_to_gpu(batch)
                        #     with torch.no_grad():
                        #         output = model(**batch, use_rl=True)
                        #     valid_reward += output.bl_reward.item()
                        #     valid_loss += output.loss.item()
                        #     if (j+1)%args.refresh_step==0:
                        #         valid_watcher.set_postfix(reward='{:.3f}'.format(valid_reward/(j+1)), loss='{:.3f}'.format(valid_loss/(j+1)), refresh=True)
                        # valid_reward/=j
                        if best_reward < r1:
                            best_reward, temper = r1, 0
                            torch.save(model.state_dict(), args.model_path)
                        else:
                            temper += 1
                            if temper > args.early_stop:
                                break
                        model = model.train()
            if epoch == args.beg_rl:
                torch.save(model.state_dict(), args.model_path+'_pt')
            if temper > args.early_stop:
                break

    else:
        valid_set = DialogSet(args, args.test_data, tokenizer)

        model.load_state_dict(torch.load(args.model_path))
        model = model.cuda()
        model = model.eval()

        predictions = []
        log = []

        if args.given:
            valid_set.training = True
            valid_loader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size)
            valid_watcher = tqdm(enumerate(valid_loader))
            for j, batch in valid_watcher:
                batch, target_indices = batch
                batch = batch_to_gpu(batch)
                with torch.no_grad():
                    output = model(**batch, use_rl=True)
                pred_logits = output.pred_logits.detach()
                for x, y in zip(pred_logits, target_indices):
                    sort_indices = torch.argsort(x, descending=True).tolist()
                    rank = [100]
                    for _y in y:
                        rank.append(sort_indices.index(_y))
                    log.append(min(rank))
        else:
            valid_loader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn_4t, batch_size=args.batch_size)
            valid_watcher = tqdm(enumerate(valid_loader))
            for i, batch in valid_watcher:
                batch = batch_to_gpu(batch)
                output_ids = model.generate(batch['input_ids'], attention_mask=batch['attention_mask'],
                                            num_beams=args.num_beams, max_length=args.max_length,
                                            min_length=args.min_length, early_stopping=True,
                                            num_return_sequences=args.num_return_sequences)
                b_pred = []
                j = 0
                for x in tokenizer.batch_decode(output_ids, skip_special_tokens=True):
                    b_pred.append(x.strip().lower())
                    if len(b_pred) == args.num_return_sequences:
                        b_pred = filter_repeat_kp(b_pred)
                        predictions.append(b_pred)
                        rank = [100]
                        for y in batch['targets'][j]:
                            y = y.lower()
                            if y in b_pred:
                                rank.append(b_pred.index(y))
                        log.append(min(rank))
                        b_pred = []
                        j+=1
            with open(args.output_path, 'w') as file:
                json.dump(predictions, file)
        log = np.array(log)
        for i in range(5):
            print('R@{}'.format(i+1), np.mean(log<i+1))
