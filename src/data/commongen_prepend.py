import os
import time
import pickle
import json
import logging
import torch
import copy
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any, Union, Optional, Tuple
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase 
from transformers import AutoTokenizer
from dataclasses import dataclass
import random
import spacy    
import re
from .utils import _torch_collate_batch, load_json

nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorCommongen:
    tokenizer: PreTrainedTokenizerBase
    qformer_tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        return self.torch_call(features)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        input_ids = [e['input_ids'] for e in examples]

        batch = {
            "input_ids": _torch_collate_batch(input_ids, self.tokenizer)
        }

        attention_mask = (batch['input_ids'] != self.tokenizer.pad_token_id)
        batch['attention_mask'] = attention_mask.long()

        batch["labels"] = _torch_collate_batch([e['labels'] for e in examples], self.tokenizer)
        attention_mask = (batch['labels'] != self.tokenizer.pad_token_id)
        batch['label_attention_mask'] = attention_mask.long()

        word_mask = batch["labels"] != self.tokenizer.pad_token_id
        batch["labels"][~word_mask] = -100

        return batch

class CommongenDataset(Dataset):
    def __init__(
        self, 
        args,
        tokenizer,
        qformer_tokenizer=None,
        split="train",
        ):
        
        self.args = args
        self.tokenizer = tokenizer
        self.split = split

        if split in ["val", "dev"]:
            split = "dev"

        start = time.time()

        logger.info(f"Creating examples from dataset file at {args.data_dir}")
        input_file = os.path.join(args.data_dir, "commongen.{}.src_alpha.txt".format(split))
        target_file = os.path.join(args.data_dir, "commongen.{}.tgt.txt".format(split))

        with open(input_file, "r") as f:
            input_datas = f.readlines()
        with open(target_file, "r") as f:
            target_datas = f.readlines()

        grounding_text = load_json(os.path.join(args.data_dir, "grounding_from_bing_text.json"))
        concepts2texts = {}
        for d in tqdm(grounding_text, desc="preparing grounded texts"):
            concepts = d["concepts"]
            concepts2texts[concepts] = []
            texts = d["bing_results"]
            for i, text in enumerate(texts[:args.text_num]):
                key = "{}_{}".format(concepts, i).encode()
                title = re.sub(r'http\S+|\S+.com', '', text["Title"])
                description = re.sub(r'http\S+|\S+.com', '', text["Description"])
                text = "title is {}, content is {}".format(title, description)
                concepts2texts[concepts].append(text)

        self.examples = []
        if args.debug_mode:
            input_datas = input_datas[:200]
            target_datas = target_datas[:200]

        concept_set = set()
        for i, (input_data, target_data) in tqdm(enumerate(zip(input_datas, target_datas))):
            input_data = input_data.strip()
            target_data = target_data.strip()
            qid = "{}_{}".format(split, i)

            if split != "train":
                if input_data in concept_set:
                    continue
                else:
                    concept_set.add(input_data)

            source = ", ".join(input_data.split(" "))
            concepts = ", ".join(sorted(input_data.split(" ")))

            target = target_data

            res = {
                "inputs": source,
                "labels": target,
                "texts": " ".join(concepts2texts.get(concepts, [])),
            }
            self.examples.append(res)

        del concepts2texts

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        item = self.examples[i].copy()

        input_tokens = self.tokenizer.tokenize(item["inputs"])
        a = self.tokenizer.tokenize("context: ")
        b = self.tokenizer.tokenize("concepts: ")
        text_ra_tokens = self.tokenizer.tokenize(item["texts"])
        text_ra_tokens = text_ra_tokens[:512-len(input_tokens)-len(a)-len(b)-self.args.prompt_length - 1]

        input_tokens = a + text_ra_tokens + b + input_tokens + [self.tokenizer.eos_token]

        item_ = {
            # "input_ids": self.tokenizer(inputs, max_length=512,  return_tensors="pt", padding="longest", truncation=True)["input_ids"],
            "input_ids": self.tokenizer.convert_tokens_to_ids(input_tokens),
            "labels": self.tokenizer(item["labels"], max_length=64)["input_ids"],
        }

        return item_
