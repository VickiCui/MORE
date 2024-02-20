import json
import os
import jsonlines

caption_dir = "datas/captions"

def load_json(fn):
    with open(fn, "r") as f:
        datas = json.load(f)
    return datas

if __name__ == "__main__":
    all_captions = {
        "image_captions":{
            "Flickr30k": [],
            "COCO": [],
            "Conceptual_Captions": [],
            "MNLI": [],
            "SNLI": []
        },
        "video_captions":{
            "LSMDC": [],
            "ActivityNet": [],
            "VATEX": [],
        }  
    }

    if os.path.exists("./datas/captions/all_captions.json"):
        data = load_json("./datas/captions/all_captions.json")
        for kk, kv in data.items():
            for k, v in kv.items():
                all_captions[kk][k] = v

    # Flickr30k
    if len(all_captions["image_captions"]["Flickr30k"]) == 0:
        this_datas = []
        for split in ["train", "val", "test"]:
            fname = os.path.join(caption_dir, "Flickr30k", "{}.json".format(split))
            datas = load_json(fname)
            datas = [d["caption"].strip() for d in datas]
            this_datas.extend(datas)
        all_captions["image_captions"]["Flickr30k"] = this_datas
    print("get {} captions for Flickr30k".format(len(all_captions["image_captions"]["Flickr30k"])))
    print("examples:")
    for d in all_captions["image_captions"]["Flickr30k"][:3]:
        print(d)

    # COCO
    if len(all_captions["image_captions"]["COCO"]) == 0:
        this_datas = []
        for split in ["train", "val", "test"]:
            fname = os.path.join(caption_dir, "COCO", "{}.json".format(split))
            datas = load_json(fname)
            datas = [d["caption"].strip() for d in datas]
            this_datas.extend(datas)
        all_captions["image_captions"]["COCO"] = this_datas
    print("get {} captions for COCO".format(len(all_captions["image_captions"]["COCO"])))
    print("examples:")
    for d in all_captions["image_captions"]["COCO"][:3]:
        print(d)
        

    # Conceptual_Captions
    if len(all_captions["image_captions"]["Conceptual_Captions"]) == 0:
        this_datas = []
        for split in ["train", "valid"]:
            fname = os.path.join(caption_dir, "Conceptual_Captions", "{}.tsv".format(split))
            with open(fname, "r") as f:
                datas = f.readlines()
            datas = [d.split("\t")[0].strip() for d in datas]
            this_datas.extend(datas)
        all_captions["image_captions"]["Conceptual_Captions"] = this_datas
    print("get {} captions for Conceptual_Captions".format(len(all_captions["image_captions"]["Conceptual_Captions"])))
    print("examples:")
    for d in all_captions["image_captions"]["Conceptual_Captions"][:3]:
        print(d)
        

    # ActivityNet
    if len(all_captions["video_captions"]["ActivityNet"]) == 0:
        this_datas = []
        for split in ["train", "val_1", "val_2"]:
            fname = os.path.join(caption_dir, "ActivityNet", "{}.json".format(split))
            datas = load_json(fname)
            for k,v in datas.items():
                for s in v["sentences"]:
                    this_datas.append(s.strip())
        all_captions["video_captions"]["ActivityNet"] = this_datas
    print("get {} captions for ActivityNet".format(len(all_captions["video_captions"]["ActivityNet"])))
    print("examples:")
    for d in all_captions["video_captions"]["ActivityNet"][:3]:
        print(d)
        


    # VATEX
    if len(all_captions["video_captions"]["VATEX"]) == 0:
        this_datas = []
        for split in ["vatex_training_v1.0", "vatex_validation_v1.0", "vatex_public_test_english_v1.1"]:
            fname = os.path.join(caption_dir, "VATEX", "{}.json".format(split))
            datas = load_json(fname)
            for d in datas:
                for s in d["enCap"]:
                    this_datas.append(s.strip())
        all_captions["video_captions"]["VATEX"] = this_datas
    print("get {} captions for VATEX".format(len(all_captions["video_captions"]["VATEX"])))
    print("examples:")
    for d in all_captions["video_captions"]["VATEX"][:3]:
        print(d)

    # LSMDC
    if len(all_captions["video_captions"]["LSMDC"]) == 0:
        this_datas = []
        fname = os.path.join(caption_dir, "LSMDC", "annotations-someone.csv")
        with open(fname, "r") as f:
            datas = f.readlines()
        datas = [d.split("\t")[1].strip() for d in datas]
        this_datas.extend(datas)
        all_captions["video_captions"]["LSMDC"] = this_datas
    print("get {} captions for LSMDC".format(len(all_captions["video_captions"]["LSMDC"])))
    print("examples:")
    for d in all_captions["video_captions"]["LSMDC"][:3]:
        print(d)
        
    # MNLI
    if len(all_captions["image_captions"]["MNLI"]) == 0:
        this_datas = []
        for split in ["multinli_1.0_train.jsonl", "multinli_1.0_dev_matched.jsonl", "multinli_1.0_dev_mismatched.jsonl"]:
            fname = os.path.join(caption_dir, "MNLI", split)
            with open(fname, "r", encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    this_datas.append(item["sentence1"].strip())
                    this_datas.append(item["sentence2"].strip())
        all_captions["image_captions"]["MNLI"] = this_datas
    print("get {} captions for MNLI".format(len(all_captions["image_captions"]["MNLI"])))
    print("examples:")
    for d in all_captions["image_captions"]["MNLI"][:3]:
        print(d)

    # SNLI
    if len(all_captions["image_captions"]["SNLI"]) == 0:
        this_datas = []
        for split in ["train", "dev", "test"]:
            fname = os.path.join(caption_dir, "SNLI", "snli_1.0_{}.jsonl".format(split))
            with open(fname, "r", encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    this_datas.append(item["sentence1"].strip())
                    this_datas.append(item["sentence2"].strip())
        all_captions["image_captions"]["SNLI"] = this_datas
    print("get {} captions for SNLI".format(len(all_captions["image_captions"]["SNLI"])))
    print("examples:")
    for d in all_captions["image_captions"]["SNLI"][:3]:
        print(d)

    with open("./datas/captions/all_captions.json", "w") as f:
        json.dump(all_captions, f)
    print("write all captions to ./datas/captions/all_captions.json")