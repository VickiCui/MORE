# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
import numpy as np
from .utils import InputPrompts
import logging

class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.config.n_passages = args.text_num
        self.config.prompt_length = args.prompt_length
        self.wrap_encoder(config=self.config)

        if args.freeze_lm:
            for name, param in self.named_parameters():
                if "prompt" not in name:
                    param.requires_grad = False
                # param.data = param.data.bfloat16()
            super().eval()
            logging.info("freeze language model")

        train_param = 0
        all_param = 0
        module_param = {}
        print("====== tuning ======")
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad == True:
                train_param += param.numel()
                print(name)
                module_name = name.split(".")[0]
                if module_name not in module_param:
                    module_param[module_name] = 0
                module_param[module_name] += param.numel()
        print("====================")

        logging.info('total param is {}'.format(all_param)) # 9860105
        logging.info('train param is {}'.format(train_param)) # 9860105
        for k, v in module_param.items():
            logging.info('train {} param is {}'.format(k, v)) # 9860105

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, labels=None, label_attention_mask=None, cands=None, **model_inputs):
        if input_ids is not None:
            input_ids = input_ids.view(input_ids.size(0), -1)
        # if attention_mask is not None:
        #     if attention_mask.dim() == 2: # beam*bsz, pass*len
        #         attention_mask = attention_mask.view(attention_mask.size(0), self.config.n_passages, -1) # beam*bsz, pass, len
            prompt_atts = torch.ones([attention_mask.size(0), self.config.n_passages, self.config.prompt_length], dtype=torch.long).to(attention_mask.device)
            attention_mask = torch.cat([prompt_atts, attention_mask], dim=-1)
        attention_mask = attention_mask.view(attention_mask.size(0), -1)

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **model_inputs,
        )

        return outputs

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, labels=None, label_attention_mask=None, **generate_kwargs):
        # self.encoder.n_passages = input_ids.size(1)
        bsz = input_ids.shape[0]
        input_ids = input_ids.view(bsz, -1)

        prompt_atts = torch.ones([bsz, self.config.n_passages, self.config.prompt_length], dtype=torch.long).to(input_ids.device)
        attention_mask = torch.cat([prompt_atts, attention_mask], dim=-1)
        attention_mask = attention_mask.view(bsz, -1)


        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def wrap_encoder(self, config, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, config=config, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder(config=self.config)

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, config, use_checkpoint=False):
        super().__init__()

        self.main_input_name = encoder.main_input_name
        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)
        self.encoder.n_passages = config.n_passages
        self.prompt_length = config.prompt_length

        self.prompt_tokens = InputPrompts(config.prompt_length, config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, return_dict=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.encoder.n_passages
        input_ids = input_ids.view(bsz*self.encoder.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.encoder.n_passages, -1)

        device = input_ids.device
        prompt_embeds = self.prompt_tokens(bsz*self.encoder.n_passages, device)

        inputs_embeds = self.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        kwargs["inputs_embeds"] = inputs_embeds

        outputs = self.encoder(input_ids=None, attention_mask=attention_mask, **kwargs)

        # outputs = (outputs[0].view(bsz, self.encoder.n_passages*passage_length, -1), ) + outputs[1:]

        if not return_dict:
            outputs[0] = outputs[0].view(bsz, self.encoder.n_passages*(passage_length+self.prompt_length), -1)
        else:
            outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.encoder.n_passages*(passage_length+self.prompt_length), -1)
        
        return outputs

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block

def cross_attention_forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output