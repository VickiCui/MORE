from bs4 import BeautifulSoup
import requests
import re
import os
import argparse
import tqdm
from src.grounding_text.process_captions import process, load_json
import json
from multiprocessing import Pool
import threading
import re

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

# maybe need to change some values
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'close',
    'cookie': '<REPLACE_TO_YOUR_COOKIE>'
          }

def get_bing_url(keywords):
    keywords = keywords.strip('\n')
    bing_url = re.sub(r'^', 'https://cn.bing.com/search?q=', keywords)
    bing_url = re.sub(r'\s', '+', bing_url)
    return bing_url

def query_bing(new_key, pages=2):
    l = []
    title_set = set()
    title_idx = []

    bing_url = get_bing_url(new_key)

    for i in range(0,10*pages,10):
        o = {}
        url = bing_url + "&rdr=1&first={}&FORM=PERE&sp=-1&lq=0&pq=&sc=0-0&qs=n&sk=&ghacc=0".format(i+1)
        # print(url)
        try:
            content = requests.get(url=url, timeout=30, headers=headers)
            soup = BeautifulSoup(content.text, 'html.parser')

            completeData = soup.find_all("li",{"class":"b_algo"})
            for i in range(0, len(completeData)):
                try:
                    o = {}
                    o["Title"] = completeData[i].h2.a.text.strip()
                    # o["link"]=completeData[i].find("a").get("href")
                    o["Description"]=completeData[i].find("div", {"class":"b_caption"}).text.strip()
                    if o["Description"] == "":
                        o["Description"]=completeData[i].find("p").text
                    o["Description"] = o["Description"].split("\xa0· ")[-1].strip()

                    if o["Title"] not in title_set:
                        title_set.add(o["Title"])
                        title_idx.append(o["Title"])
                    else:
                        idx = title_idx.index(o["Title"])
                        if l[idx]["Description"] == o["Description"]:
                            continue
                    if not is_contains_chinese(o["Title"] + " " + o["Description"]):
                        o["Title"] = re.sub(r'http\S+|\S+.com', '', o["Title"])
                        o["Description"] = re.sub(r'http\S+|\S+.com', '', o["Description"])
                        l.append(o)
                    # print(o)
                except:
                    pass
        except:
            pass
    return l

def get_commongen_data(data_dir="datas/commongen", splits=["train", "dev", "test"]):
    all_data = []
    for split in splits:
        fname = os.path.join(data_dir, "commongen.{}.src_alpha.txt".format(split))
        with open(fname, "r") as f:
            datas = f.readlines()
        for i, data in enumerate(datas):
            d = {}
            data = data.strip()
            d["id"] = "{}_{}".format(split, i)
            d["sent"] = ", ".join(sorted(data.split(" ")))
            all_data.append(d)
    return all_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", default="./datas/commongen", type=str)

    parser.add_argument(
        "--save_dir", default="./datas/commongen", type=str)

    parser.add_argument(
        "--n_threads", default=1, type=int)

    parser.add_argument(
        "--pages", default=2, type=int)

    args = parser.parse_args()


    all_data = get_commongen_data(args.data_dir)
    print("data loaded")

    querys = set()
    for d in all_data:
        querys.add(d["sent"])

    output_data_ = []
    if os.path.exists(os.path.join(args.save_dir, "bing_text_for_commongen.json")):
        with open(os.path.join(args.save_dir, "bing_text_for_commongen.json"), "r") as f:
            output_data_ = json.load(f)
    c2i = {}
    for i, d in enumerate(output_data_):
        if len(d["bing_results"]) > 0:
            c2i[d["concepts"]] = i

    def func(query):
        if query not in c2i:
            res = query_bing(query, pages=args.pages)

            return {
                "concepts": query,
                "bing_results": res,
                }
        else:
            return output_data_[c2i[query]]

    print("searching...")
    output_data = []
    if args.n_threads == 1:
        for query in tqdm.tqdm(querys):
            output_data.append(func(query))
    else:
        with Pool(args.n_threads) as p:
            output_data = list(tqdm.tqdm(p.imap(func, querys), total=len(querys)))

    output_path = os.path.join(args.save_dir, "bing_text_for_commongen.json")
    with open(output_path, 'w') as fi:
        json.dump(output_data, fi)
