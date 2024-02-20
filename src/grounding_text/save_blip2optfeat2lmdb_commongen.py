import json
import os
import sys 
from tqdm import tqdm
import jsonlines
import torch
import argparse
import shutil
from PIL import Image
import lmdb
import pickle
from transformers import Blip2Processor, Blip2Model
import re

def get_files(file_dir):
#遍历filepath下所有文件，忽略子目录
    all_file = []
    files = os.listdir(file_dir)
    for fi in files:
        fi_d = os.path.join(file_dir,fi)            
        if os.path.isdir(fi_d):
            continue             
        else:
            all_file.append(fi_d)
            
    return all_file

def get_data(data_dir="datas/commongen", splits=["train", "dev", "test"]):
    all_data = []
    for split in splits:
        fname = os.path.join(data_dir, "commongen.{}.src_alpha.txt".format(split))
        with open(fname, "r") as f:
            datas = f.readlines()
        for i, data in enumerate(datas):
            d = {}
            data = data.strip()
            d["id"] = "{}_{}".format(split, i)
            d["sent"] = ", ".join(data.split(" "))
            all_data.append(d)
    return all_data

def get_feature_from_bing(data_dir, save_path, model_path, batch_size=32, debug_mode=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2Model.from_pretrained(
        model_path, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
    )

    model = model.to(device)
    model.eval()

    env = lmdb.open(save_path, readonly=False, create=True, map_size=4 * 1024**4)
    txn = env.begin(write=True)

    with open(os.path.join(data_dir, "bing_text_for_commongen.json"), "r") as f:
        bing_text = json.load(f)
    if debug_mode:
        bing_text = bing_text[:20]


    with torch.no_grad():
        s = 0
        cnt = 0
        pbar = tqdm(total=len(bing_text))
        while s < len(bing_text):
            ds = bing_text[s:s+batch_size]
            s += batch_size

            keys = []
            texts = []
            # laod a batch of text
            for d in ds:
                concepts = d["concepts"]
                texts = d["bing_results"]
                for i, text in enumerate(texts):
                    key = "{}_{}".format(concepts, i).encode()
                    c = txn.get(key)
                    if c is None:
                        title = re.sub(r'http\S+|\S+.com', '', text["Title"])
                        description = re.sub(r'http\S+|\S+.com', '', text["Description"])
                        text = "title is {}, content is {}".format(title, description)
                        texts.append(text)
                        keys,append(key)
                    else:
                        pass
            # get features
            if len(texts) is not None:
                inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
                text_features = model.language_model.get_input_embeddings()(inputs["input_ids"]).cpu()

                for key, text_feature, attention_mask in zip(keys, text_features, inputs["attention_mask"]):
                    l = torch.sum(attention_mask)
                    text_feature = text_feature[:l]
                    text_feature = text_feature.squeeze(0)
                    value = {"text_feature": text_feature}
                    value = pickle.dumps(value)

                    txn.put(key, value)
                    cnt += 1
                    if cnt % 1000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)

            pbar.update(batch_size)

            
        txn.commit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", default="./datas/commongen", type=str)
    parser.add_argument(
        "--save_path", default="./datas/commongen/blip2opt_text_feats.lmdb", type=str)
    parser.add_argument(
        "--model_path", default="", type=str)
    parser.add_argument(
        "--batch_size", default=32, type=int)
    parser.add_argument(
        "--debug_mode",  action='store_true')

    args = parser.parse_args()

    get_feature_from_bing(args.data_dir, args.save_path, args.model_path, args.batch_size, debug_mode=args.debug_mode)