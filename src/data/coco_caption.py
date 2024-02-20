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
import lmdb
import clip
import random
import spacy    
from .utils import get_image_input, get_graph_input, get_text_input, _torch_collate_batch, load_json

nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorCoco:
    tokenizer: PreTrainedTokenizerBase

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

        # batch["raw_text_input"] = [e['raw_text_input'] for e in examples]
        ################# image ##################
        batch["image_inputs"] = None
        if examples[0].get("image_inputs", None) is not None:
            image_inputs = [e['image_inputs'] for e in examples]
            image_inputs = torch.stack(image_inputs) # (B,n,3,224,224)
            batch["image_inputs"] = image_inputs

        batch["image_embeds"] = None
        if examples[0].get("image_embeds", None) is not None:
            image_embeds = [e['image_embeds'] for e in examples]
            image_embeds = torch.stack(image_embeds) # (B,nxl,d)
            batch["image_embeds"] = image_embeds
            

        return batch

class Lmdb():
    def __init__(self, lmdb_file, readonly=True):
        if readonly:
            self.env = lmdb.open(lmdb_file, lock=False,
                                 readonly=True, create=False)
            self.txn = self.env.begin(buffers=True)
        else:
            # prepro
            self.env = lmdb.open(lmdb_file, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)

    def get(self, image_id):
        key = str(image_id).encode()
        c = self.txn.get(key)
        if not c:
            return None
        image_info = pickle.loads(c)
        return image_info

    def exist(self, image_id):
        key = str(image_id).encode()
        c = self.txn.get(key)
        if not c:
            return False
        else:
            return True

    def get_keys(self):
        keys = []
        for k,v in self.txn.cursor():
            keys.append(bytes(k).decode("utf-8"))
        return keys

    def __del__(self):
        self.env.close()


class CocoDataset(Dataset):
    def __init__(
        self, 
        args,
        tokenizer,
        split="train",
        ):
        
        self.args = args
        self.tokenizer = tokenizer
        self.split = split

        if split in ["val", "dev"]:
            split = "val"

        clip_model, self.image_process = clip.load(args.image_model_path, device="cpu")

        logger.info(f"Creating examples from dataset file at {args.data_dir}")

        self.examples = []

        with open(os.path.join(args.data_dir, split+".json"), "r") as f:
            datas = json.load(f)

        if args.debug_mode:
            datas = datas[:100]

        for data in tqdm(datas):
            res = {
                "input_ids": tokenizer("describes", max_length=256)["input_ids"],
                "labels": tokenizer(data["caption"].lower().strip(), max_length=256)["input_ids"],
                "image_files": [data["filename"]],
            }
            self.examples.append(res)

        del clip_model

        self.image_lmdb = None

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        item = self.examples[i].copy()

        if self.image_lmdb is None:
            self.image_lmdb = Lmdb(self.args.image_lmdb, readonly=True)

        item["image_inputs"] = None
        image_files = item["image_files"]
        if self.image_lmdb is not None:
            image_embeds = [self.image_lmdb.get(str(image_file))["image_feature"] for image_file in image_files] 
            item["image_embeds"] = torch.cat(image_embeds, dim=0)
        else:
            item["image_inputs"] = get_image_input(self.image_process, image_files)

        return item
