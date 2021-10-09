import json
import re

import scipy
import torch
from parlai.core.metrics import MacroAverageMetric, F1Metric
import numpy as np


def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()  # entity全称，如&gt;
        key = sz.group('name')  # 去除&;后entity,如&gt;为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            # 以空串代替
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr

def evaluate_uni_f1(predictions, references):
    assert len(predictions) == len(references)
    id = 0
    scores = {}
    for pred, ref in zip(predictions, references):
        scores[id] = F1Metric.compute(pred, [ref])
        id += 1
    metric = MacroAverageMetric(scores)
    return metric.value(), scores

def evaluate(predictions, references):
    precision_list, recall_list, f1_list = [], [], []
    for prediction, reference in zip(predictions, references):
        scores = run_classic_metrics(
            [1 if item in reference else 0 for item in prediction],
            prediction, reference, ['precision', 'recall', 'f_score'], ['M']
        )
        precision_list.append(scores['precision@M'])
        recall_list.append(scores['recall@M'])
        f1_list.append(scores['f_score@M'])
    return macro_averaged_score(precision_list, recall_list), f1_list

def run_classic_metrics(match_list, pred_list, tgt_list, score_names, topk_range, type='exact'):
    """
    Return a dict of scores containing len(score_names) * len(topk_range) items
    score_names and topk_range actually only define the names of each score in score_dict.
    :param match_list:
    :param pred_list:
    :param tgt_list:
    :param score_names:
    :param topk_range:
    :param type: exact or partial
    :return:
    """
    score_dict = {}
    if len(tgt_list) == 0:
        for topk in topk_range:
            for score_name in score_names:
                score_dict['{}@{}'.format(score_name, topk)] = 0.0
        return score_dict

    assert len(match_list) == len(pred_list)
    for topk in topk_range:
        if topk == 'k':
            cutoff = len(tgt_list)
        elif topk == 'M':
            cutoff = len(pred_list)
        else:
            cutoff = topk

        if len(pred_list) > cutoff:
            pred_list_k = np.asarray(pred_list[:cutoff])
            match_list_k = match_list[:cutoff]
        else:
            pred_list_k = np.asarray(pred_list)
            match_list_k = match_list

        if type == 'partial':
            cost_matrix = np.asarray(match_list_k, dtype=float)
            if len(match_list_k) > 0:
                # convert to a negative matrix because linear_sum_assignment() looks for minimal assignment
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(-cost_matrix)
                match_list_k = cost_matrix[row_ind, col_ind]
                overall_cost = cost_matrix[row_ind, col_ind].sum()
            '''
            print("\n%d" % topk)
            print(row_ind, col_ind)
            print("Pred" + str(np.asarray(pred_list)[row_ind].tolist()))
            print("Target" + str(tgt_list))
            print("Maximum Score: %f" % overall_cost)
            print("Pred list")
            for p_id, (pred, cost) in enumerate(zip(pred_list, cost_matrix)):
                print("\t%d \t %s - %s" % (p_id, pred, str(cost)))
            '''

        # Micro-Averaged Method
        correct_num = int(sum(match_list_k))
        # Precision, Recall and F-score, with flexible cutoff (if number of pred is smaller)
        micro_p = float(sum(match_list_k)) / float(len(pred_list_k)) if len(pred_list_k) > 0 else 0.0
        micro_r = float(sum(match_list_k)) / float(len(tgt_list)) if len(tgt_list) > 0 else 0.0

        if micro_p + micro_r > 0:
            micro_f1 = float(2 * (micro_p * micro_r)) / (micro_p + micro_r)
        else:
            micro_f1 = 0.0
        # F-score, with a hard cutoff on precision, offset the favor towards fewer preds
        micro_p_hard = float(sum(match_list_k)) / cutoff if len(pred_list_k) > 0 else 0.0
        if micro_p_hard + micro_r > 0:
            micro_f1_hard = float(2 * (micro_p_hard * micro_r)) / (micro_p_hard + micro_r)
        else:
            micro_f1_hard = 0.0

        for score_name, v in zip(['correct', 'precision', 'recall', 'f_score', 'precision_hard', 'f_score_hard'], [correct_num, micro_p, micro_r, micro_f1, micro_p_hard, micro_f1_hard]):
            score_dict['{}@{}'.format(score_name, topk)] = v

    # return only the specified scores
    return_scores = {}
    for topk in topk_range:
        for score_name in score_names:
            return_scores['{}@{}'.format(score_name, topk)] = score_dict['{}@{}'.format(score_name, topk)]

    return return_scores

def macro_averaged_score(precisionlist, recalllist):
    precision = np.average(precisionlist)
    recall = np.average(recalllist)
    f_score = 0
    if(precision or recall):
        f_score = round((2 * (precision * recall)) / (precision + recall), 4)
    return precision, recall, f_score

def filter_repeat_mention(mentions):
    # filter repeat mention while keep order
    new_mention_list = []
    for mention in mentions:
        mention = re.sub('<.+?>', '', mention).strip()
        if mention not in new_mention_list and len(mention)>0:
            new_mention_list.append(mention)
    return new_mention_list

def filter_repeat_kp(x_list):
    x_set = set()
    y_list = []
    for x in x_list:
        if x not in x_set:
            x_set.add(x)
            y_list.append(x)
    return y_list

def len_list_to_mask(len_list):
    if isinstance(len_list, list):
        len_list = torch.tensor(len_list)
    max_len = len_list.max()
    bsz = len_list.shape[0]
    mask = torch.arange(max_len).expand(bsz, max_len) < len_list.unsqueeze(1)
    return mask

def flat_list(input):
    output = []
    for x in input:
        output += x
    return output

def batch_to_gpu(batch):
    for k in batch:
        if batch[k] is not None:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].cuda()
    return batch

# def evaluate(predictions, references):
#     f1_scores = {}
#     id = 0
#     for pred, ref in zip(predictions, references):
#         f1_scores[id]=F1Metric.compute(pred, [ref])
#         id += 1
#     return MacroAverageMetric(f1_scores).value(), f1_scores

def load_json_file(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data