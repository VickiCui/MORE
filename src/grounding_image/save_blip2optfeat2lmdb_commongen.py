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

def get_files(file_dir):
#遍历filepath下所有文件，忽略子目录
    all_file = []
    files = os.listdir(file_dir)
    for fi in files:
        fi_d = os.path.join(file_dir,fi)            
        if os.path.isdir(fi_d):
            continue             
        else:
            all_file.append(fi)
            
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

    all_datas = get_data(data_dir)
    if debug_mode:
        all_datas = all_datas[:20]
    image_dir = os.path.join(data_dir, "bing_images")

    with torch.no_grad():
        s = 0
        cnt = 0
        pbar = tqdm(total=len(all_datas))
        while s < len(all_datas):
            ds = all_datas[s:s+batch_size]
            s += batch_size

            keys = []
            images = []
            # laod a batch of image
            for d in ds:
                image_dir_ = d["sent"]
                image_files = get_files(os.path.join(image_dir, image_dir_))
                for image_file in image_files:
                    key = os.path.join(image_dir_, image_file).encode()
                    c = txn.get(key)
                    if c is None:
                        try:
                            image = Image.open(os.path.join(image_dir, image_dir_, image_file))
                            keys.append(key)
                            images.append(image)
                        except:
                            pass
                    else:
                        pass
            # get features
            if len(images) is not None:
                inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
                image_features = model.get_qformer_features(**inputs).cpu()

                for key, image_feature in zip(keys, image_features):
                    image_feature = image_feature.squeeze(0)
                    value = {"image_feature": image_feature}
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
        "--save_path", default="./datas/commongen/blip2opt_image_feats.lmdb", type=str)
    parser.add_argument(
        "--model_path", default="", type=str)
    parser.add_argument(
        "--batch_size", default=32, type=int)
    parser.add_argument(
        "--debug_mode",  action='store_true')

    args = parser.parse_args()

    get_feature_from_bing(args.data_dir, args.save_path, args.model_path, args.batch_size, debug_mode=args.debug_mode)