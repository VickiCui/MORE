"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import AutoConfig
import numpy as np

# from .registry import registry
from transformers import PreTrainedModel
from .modeling_t5 import T5Config, T5ForConditionalGeneration
from .integrator import BertModel
from src.model.blip2_qformer import Blip2Qformer
from .utils import LayerNorm, InputPrompts

# @registry.register_model("ke_t5")
class BaseKnowledgeEnhancedT5(PreTrainedModel):
    def __init__(
        self,
        config,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__(config)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def maybe_autocast(self, dtype=torch.float32):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def _normalize(self, feature, dim=-1):
        norm = feature.norm(p=2, dim=dim, keepdim=True)
        feature = feature.div(norm + 1e-8)
        return feature

    def init_Qformer(self, integrator_path, num_query_token, cross_attention_freq=1, num_integrator_layers=2, integrator_attn_list=[["self", "cross"], ["self", "cross"]], num_former_layers=1, former_attn_list=[["cross"]]):
        integrator_embedding = Blip2Qformer(
            vision_path="./pre_trained_lm/eva_vit_g.pth",
            former_path="./pre_trained_lm/bert-base-uncased",
        )
        msg = integrator_embedding.load_from_pretrained("./pre_trained_lm/blip2_pretrained.pth")
        integrator_embedding.eval()
        integrator_embedding = integrator_embedding.Qformer.bert.embeddings
        for name, param in integrator_embedding.named_parameters():
            param.requires_grad = False

        integrator_config = AutoConfig.from_pretrained(integrator_path)
        integrator_config.hidden_size = 768
        integrator_config.encoder_width = integrator_config.hidden_size
        # insert cross-attention layer every other block
        # integrator_config.query_length = num_query_token
        integrator_config.num_hidden_layers = num_integrator_layers
        integrator = BertModel(integrator_config, attn_list=[["self", "cross"], ["self", "cross"]])
        integrator.apply(self._init_weights)

        former_config = AutoConfig.from_pretrained(integrator_path)
        former_config.hidden_size = 768
        former_config.encoder_width = former_config.hidden_size
        # former_config.query_length = num_query_token
        former_config.num_hidden_layers = num_former_layers
        former = BertModel(former_config, attn_list=[["cross"]])
        former.apply(self._init_weights)

        query_tokens = InputPrompts(num_query_token, former_config.hidden_size)

        t5_proj = nn.Linear(
            former_config.hidden_size, self.language_model.config.hidden_size
        )
        
        t5_proj.apply(self._init_weights)
           
        return integrator_embedding, integrator, former, query_tokens, t5_proj

    def forward_image(
            self,
            image_inputs=None,
            image_inputs_attention_mask=None,
            device=None,
        ):
        image_embeds, image_atts = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

        if image_inputs is not None and self.num_query_token>0:
            with self.maybe_autocast():
                bsz, num, l, hid = image_inputs.size()
                image_embeds = image_inputs.view(bsz, -1, hid)

                if self.image_proj is not None:
                    image_embeds = self.image_proj(image_embeds) # (bsz, N*l, h)
                image_atts = image_inputs_attention_mask.view(bsz, -1)

        return image_embeds, image_atts

    def forward_text(
            self,
            text_inputs=None,
            text_inputs_attention_mask=None,
            device=None,
        ):
        text_embeds, text_atts = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

        if text_inputs is not None and self.num_query_token>0:
            bsz, n, l = text_inputs.size()           
            with self.maybe_autocast():
                bsz, l, hid = text_inputs.size()
                text_embeds = text_inputs.view(bsz, -1, hid)
                if self.text_proj is not None:
                    text_embeds = self.text_proj(text_embeds) # (bsz, N*l, h)
                text_atts = text_inputs_attention_mask.view(bsz, -1)
        
        return text_embeds, text_atts

    def forward_query(
            self,
            embeds=None,
            atts=None,
            former_text_input_ids=None,
            former_text_input_masks=None,
            device=None,
        ):
        retrieved_inputs_t5, retrieved_atts_t5 = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

        # print(former_text_input_ids.shape)
        if len(embeds) > 0:
            input_embeds = self.integrator_embedding(former_text_input_ids)
            # input query retrieved data
            integrator_output = self.integrator(
                query_embeds=input_embeds,
                attention_mask=former_text_input_masks,
                encoder_hidden_states=embeds,
                encoder_attention_mask=atts,
                return_dict=True,
            )

            # prompt query integrated data
            query_token_embeds = self.query_tokens(embeds.shape[0], device)
            encoder_attention_mask = torch.ones(integrator_output.last_hidden_state.size()[:-1], dtype=torch.long).to(device)
            former_output = self.former(
                query_embeds=query_token_embeds,
                encoder_hidden_states=integrator_output.last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )

            retrieved_inputs_t5 = self.t5_proj(former_output.last_hidden_state)
            retrieved_atts_t5 = torch.ones(retrieved_inputs_t5.size()[:-1], dtype=torch.long).to(device)

        return retrieved_inputs_t5, retrieved_atts_t5
        

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared


# @registry.register_model("ke_t5")
class PrepandlKnowledgeEnhancedT5(BaseKnowledgeEnhancedT5):
    def __init__(
        self,
        config,
        args,

        freeze_image_encoder=None,
        freeze_text_encoder=None,
        num_query_token=None,
        prompt_length=32,

        integrator_path=None,

        t5_model="google/flan-t5-xl",
        resume_checkpoint=None,

        cand_ids=None,
        mmd_scale=None,
        
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__(config)
        print("PrepandlKnowledgeEnhancedT5!")

        self.logits_processor = None

        self.args = args

        ########## lm ##########
        t5_config = T5Config.from_pretrained(t5_model)
        # t5_config.dense_act_fn = "gelu"
        self.language_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
        self.generation_config = self.language_model.generation_config

        if args.freeze_lm:
            for name, param in self.language_model.named_parameters():
                param.requires_grad = False
                # param.data = param.data.bfloat16()
            self.language_model.eval()
            logging.info("freeze language model")

        self.prompt_tokens = InputPrompts(prompt_length, self.language_model.config.hidden_size)

        if resume_checkpoint is not None:
            logging.info("Init pretrained model from pretrained_model_path: " + resume_checkpoint)
            print("Init pretrained model from pretrained_model_path: " + resume_checkpoint)
            model_dict = torch.load(resume_checkpoint+"/pytorch_model.bin")
            incompatible_keys = self.load_state_dict(model_dict, strict=False)
            print(incompatible_keys)

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
        
    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        labels=None,
        label_attention_mask=None,
        ):

        device = input_ids.device

        ########## prompt ##########
        prompt_embeds_t5 = self.prompt_tokens(input_ids.shape[0], device)
        prompt_atts_t5 = torch.ones(prompt_embeds_t5.size()[:-1], dtype=torch.long).to(device)

        ########## lm ##########
        with self.maybe_autocast():
            encoder_atts = torch.cat([prompt_atts_t5, attention_mask], dim=1)
            inputs_embeds = self.language_model.encoder.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prompt_embeds_t5, inputs_embeds], dim=1)

            targets = labels

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=label_attention_mask,
                return_dict=True,
                output_hidden_states=False,
                labels=targets,
            )
            loss = outputs.loss

        return {
            "loss": loss,
            # "mmd_loss":mmd_loss,
            "logits": outputs.logits,
            }

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        label_attention_mask=None,
        **generate_kwargs,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        device = input_ids.device
        
        ########## prompt ##########
        prompt_embeds_t5 = self.prompt_tokens(input_ids.shape[0], device)
        prompt_atts_t5 = torch.ones(prompt_embeds_t5.size()[:-1], dtype=torch.long).to(device)

        ########## lm ##########
        with self.maybe_autocast():
            encoder_atts = torch.cat([prompt_atts_t5, attention_mask], dim=1)
            inputs_embeds = self.language_model.encoder.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prompt_embeds_t5, inputs_embeds], dim=1)

            generate_kwargs["logits_processor"] = self.logits_processor

            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                **generate_kwargs,
            )

        return outputs