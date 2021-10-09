from torch.distributions import Categorical
from transformers import BartPretrainedModel, BartConfig
from transformers.activations import get_activation
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.bart import BartModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartEncoder, shift_tokens_right

class RaGe(nn.Module):
    def __init__(self, ranker, generator, rank_gpu_ids, gen_gpu_ids):
        super().__init__()
        self.ranker = ranker
        self.generator = generator
        self.rank_gpu_ids = rank_gpu_ids
        self.gen_gpu_ids = gen_gpu_ids
        self.device_rank = torch.device('cuda', self.rank_gpu_ids[0])
        self.device_gen = torch.device('cuda', self.gen_gpu_ids[0])

    def cuda(self, device = None):
        self.ranker = self.ranker.to(self.device_rank)
        self.generator = self.generator.to(self.device_gen)
        if len(self.rank_gpu_ids) > 1:
            self.ranker = nn.DataParallel(self.ranker, device_ids=self.rank_gpu_ids)
        if len(self.gen_gpu_ids) > 1:
            self.generator = nn.DataParallel(self.generator, device_ids=self.gen_gpu_ids)
        return self

    def to_device(self, tensor_list, target_device):
        return [x.to(target_device) for x in tensor_list]

    def forward(self, input_ids, attention_mask, kn_input_ids, kn_attention_mask, decoder_input_ids, labels, rl=True):
        bsz, ksz = kn_input_ids.shape[:2]
        input_ids, attention_mask, kn_input_ids, kn_attention_mask = \
            self.to_device([input_ids, attention_mask, kn_input_ids, kn_attention_mask], self.device_rank)
        kn_mask = torch.sum(kn_attention_mask, dim=-1) > 0
        if rl:
            output_logits = self.ranker(input_ids, attention_mask, kn_input_ids, kn_attention_mask) # B K
            output_logits[~kn_mask] = -1e9
            output_logits = F.softmax(output_logits, dim=-1)

            sample_indices = Categorical(probs=output_logits).sample()
            baseli_indices = torch.max(output_logits, dim=-1)[1]
            row = torch.arange(bsz)
            sampled_kn_input_ids = kn_input_ids[row, sample_indices]
            sampled_kn_attention_mask = kn_attention_mask[row, sample_indices]
            baselin_kn_input_ids = kn_input_ids[row, baseli_indices]
            baselin_kn_attention_mask = kn_attention_mask[row, baseli_indices]
            selected_kn_input_ids = torch.cat((sampled_kn_input_ids.unsqueeze(0), baselin_kn_input_ids.unsqueeze(0)), dim=0)
            selected_kn_attention_mask = torch.cat((sampled_kn_attention_mask.unsqueeze(0), baselin_kn_attention_mask.unsqueeze(0)), dim=0)

            expand_input_ids = input_ids.unsqueeze(0).expand(2, bsz, input_ids.shape[-1])
            expand_attention_mask = attention_mask.unsqueeze(0).expand(2, bsz, input_ids.shape[-1])

            generator_input_ids = torch.cat([expand_input_ids, selected_kn_input_ids], dim=-1).reshape(2*bsz, -1)
            generator_attention_mask = torch.cat([expand_attention_mask, selected_kn_attention_mask], dim=-1).reshape(2*bsz, -1)
            generator_decoder_input_ids = decoder_input_ids.unsqueeze(0).expand(2, bsz, decoder_input_ids.shape[-1]).reshape(2*bsz, -1)
            generator_labels = labels.unsqueeze(0).expand(2, bsz, labels.shape[-1]).reshape(2*bsz, -1)

            generator_input_ids, generator_attention_mask, generator_decoder_input_ids, generator_labels = \
            self.to_device([generator_input_ids, generator_attention_mask, generator_decoder_input_ids, generator_labels], self.device_gen)

            generator_loss = self.generator(generator_input_ids, generator_attention_mask, generator_decoder_input_ids, labels=generator_labels).loss
            bsz_mean_generator_loss = (generator_loss.sum(-1)/(torch.sum(generator_labels>-1, dim=-1))).reshape(2, bsz).detach()
            rewards = bsz_mean_generator_loss[1] - bsz_mean_generator_loss[0]
            ranker_loss = -1.0*output_logits[row, sample_indices].log().to(self.device_gen)*rewards.reshape(-1)
            ranker_loss = ranker_loss.sum()/bsz
            generator_loss = generator_loss.sum()/(torch.sum(generator_labels>-1))
            return ranker_loss, generator_loss
        else:
            # dummy_logits = torch.rand_like(kn_mask.float())
            # dummy_logits[~kn_mask] = 0.
            # _,selected_indices = torch.max(dummy_logits, dim=-1)
            output_logits = self.ranker(input_ids, attention_mask, kn_input_ids, kn_attention_mask)  # B K
            output_logits[~kn_mask] = -1e9
            selected_indices = (torch.sum(kn_mask, dim=-1) > 1).long()
            ranker_loss = F.cross_entropy(output_logits, selected_indices)
            row = torch.arange(bsz)

            selected_kn_input_ids = kn_input_ids[row, selected_indices]
            selected_kn_attention_mask = kn_attention_mask[row, selected_indices]

            generator_input_ids = torch.cat([input_ids, selected_kn_input_ids], dim=-1)
            generator_attention_mask = torch.cat([attention_mask, selected_kn_attention_mask], dim=-1)
            generator_decoder_input_ids = decoder_input_ids
            generator_labels = labels

            generator_input_ids, generator_attention_mask, generator_decoder_input_ids, generator_labels = \
                self.to_device([generator_input_ids, generator_attention_mask, generator_decoder_input_ids, generator_labels], self.device_gen)

            generator_loss = self.generator(generator_input_ids, generator_attention_mask, generator_decoder_input_ids,
                                            labels=generator_labels).loss
            generator_loss = generator_loss.sum()/(torch.sum(generator_labels>-1))
            return ranker_loss, generator_loss



    def generate(self, input_ids, attention_mask, kn_input_ids, kn_attention_mask, **kwargs):
        bsz, ksz = input_ids.shape[:2]
        output_logits = self.ranker(input_ids, attention_mask, kn_input_ids, kn_attention_mask) # B K
        kn_mask = torch.sum(kn_attention_mask, dim=-1) > 0
        output_logits[~kn_mask] = -1e9
        output_logits = F.softmax(output_logits, dim=-1)
        selected_indices = torch.max(output_logits, dim=-1)[1]
        selected_kn_input_ids = kn_input_ids[torch.arange(bsz), selected_indices]
        selected_kn_attention_mask = kn_attention_mask[torch.arange(bsz), selected_indices]
        generator_input_ids = torch.cat([input_ids, selected_kn_input_ids], dim=1).to(self.device_gen)
        generator_attention_mask = torch.cat([attention_mask, selected_kn_attention_mask], dim=1).to(self.device_gen)
        return self.generator.generate(generator_input_ids, attention_mask=generator_attention_mask, **kwargs)


class RankerHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classif_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x.squeeze(-1)


class Ranker(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.ranker_head = RankerHead(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        kn_input_ids=None,
        kn_attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0][:, 0] # B H

        bsz, ksz, _ = kn_input_ids.shape
        kn_encoder_outputs = self.encoder(
            input_ids=kn_input_ids.reshape(bsz*ksz, -1),
            attention_mask=kn_attention_mask.reshape(bsz*ksz, -1),
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        kn_last_hidden_state = kn_encoder_outputs[0][:, 0].reshape(bsz, ksz, -1) # B K H

        logits = self.ranker_head(torch.cat([last_hidden_state.unsqueeze(1).expand_as(kn_last_hidden_state), kn_last_hidden_state], dim=-1)) # B K 1
        return logits # B K


class Generator(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            bsz = outputs[0].shape[0]
            masked_lm_loss = masked_lm_loss.reshape(bsz, -1)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past