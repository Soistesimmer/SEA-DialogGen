import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import ElectraModel
from transformers.activations import get_activation


class RankerHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x.squeeze(-1)


class Model(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.ranker_head = RankerHead(model.config)

    def forward(self, input, use_rl=True):
        uttera_hidden_states = self.model(input_ids=input['input_ids'], token_type_ids=input['token_type_ids'], attention_mask=input['attention_mask'])[0]
        extent_hidden_states = uttera_hidden_states[input['mapping']]
        extent_hidden_states[~input['candidate_mask'].bool()] = 0
        candid_hidden_states = torch.sum(extent_hidden_states, dim=1)
        dialog_hidden_states = uttera_hidden_states[:, 0][input['mapping']]
        logits = self.ranker_head(torch.cat([dialog_hidden_states, candid_hidden_states], dim=-1))
        candidates = torch.ones_like(input['rewards'])*-1e9
        candidates[input['rewards']>-1] = logits.to(candidates)
        pred = F.softmax(candidates, dim=-1)
        row = torch.arange(pred.shape[0])

        if use_rl:
            bl_i = pred.max(-1)[-1] # baseline indices
            sp_i = Categorical(probs=pred).sample() # sampled indices
            loss = -pred[row, sp_i].log()*(input['rewards'][row, sp_i]-input['rewards'][row, bl_i])
        else:
            tgt_i = input['rewards'].max(-1)[-1]
            loss = -pred[row, tgt_i].log()
        return pred, loss.mean()