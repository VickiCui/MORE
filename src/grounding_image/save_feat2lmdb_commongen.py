import json
import os
import sys 
from tqdm import tqdm
import jsonlines
import torch
import numpy as np
import clip as clip_folder
import argparse
import shutil
from PIL import Image
import lmdb
import pickle
from src.model.blip2_qformer import Blip2Qformer
import logging

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
            # d["sent"] = ", ".join(data.split(" "))

            dir_name = data.split(" ")
            dir_name = sorted(dir_name)
            dir_name = "#".join(dir_name)
            d["sent"] = dir_name
            all_data.append(d)
    return all_data

def get_feature_from_bing(data_dir, save_path, mode="image", debug_mode=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Blip2Qformer(
        vision_path="./pre_trained_lm/eva_vit_g.pth",
        former_path="./pre_trained_lm/bert-base-uncased",
    )
    msg = model.load_from_pretrained("./pre_trained_lm/blip2_pretrained.pth")
    logging.info(msg)

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    model = model.to(device)
    model.eval()

    vis_processors = model.get_image_processor()
    txt_processors = model.get_text_processor()

    env = lmdb.open(save_path, readonly=False, create=True, map_size=4 * 1024**4)
    txn = env.begin(write=True)

    all_datas = get_data(data_dir)
    if debug_mode:
        all_datas = all_datas[:20]
    image_dir = os.path.join(data_dir, "bing_image_for_commongen")

    with torch.no_grad():
        cnt = 0
        for d in tqdm(all_datas):
        # for d in tqdm([]):
            qid = d["id"]
            image_dir_ = d["sent"]
            image_files = get_files(os.path.join(image_dir, image_dir_))
            for image_file in image_files:
                key = image_file.encode()
                c = txn.get(key)
                if c is not None:
                    pass
                else:
                    try:
                        raw_image = Image.open(image_file).convert("RGB")
                        image = vis_processors(raw_image).unsqueeze(0).to(device)

                        sample = {"image": image}

                        if mode == "image":
                            image_feature = model.extract_features(sample, mode=mode).image_embeds.cpu()

                        if mode == "multimodal":
                            text_input = txt_processors(" ".join(d["sent"].split("#")))
                            sample["text_input"] = [text_input]
                        
                            image_feature = model.extract_features(sample, mode=mode).multimodal_embeds.cpu()


                        inputs = {"image_feature": image_feature}
                        value = pickle.dumps(inputs)

                        txn.put(key, value)
                        cnt += 1
                        if cnt % 1000 == 0:
                            txn.commit()
                            txn = env.begin(write=True)
                    except:
                        pass

        blank_image = Image.new('RGB', (256, 256), (255, 255, 255))
        image = vis_processors(blank_image).unsqueeze(0).to(device)
        sample = {"image": image}

        if mode == "image":
            image_feature = model.extract_features(sample, mode=mode).image_embeds.cpu()

        if mode == "multimodal":
            text_input = txt_processors("")
            sample["text_input"] = [text_input]
        
            image_feature = model.extract_features(sample, mode=mode).multimodal_embeds.cpu()

        inputs = {"image_feature": image_feature}
        value = pickle.dumps(inputs)
        key = "blank_image".encode()
        txn.put(key, value)

        txn.commit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", default="./datas/commongen", type=str)
    parser.add_argument(
        "--save_path", default="./datas/commongen/blip2_image_feats.lmdb", type=str)
    parser.add_argument(
        "--mode", default="image", type=str)
    parser.add_argument(
        "--debug_mode",  action='store_true')

    args = parser.parse_args()

    get_feature_from_bing(args.data_dir, args.save_path, args.mode, debug_mode=args.debug_mode)