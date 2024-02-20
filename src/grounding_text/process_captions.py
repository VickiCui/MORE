# process all_captions with Lemmatization and remove Stop Words
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import spacy
import json
import os

en_nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
# stop_words = {"should've", 'did', 'for', 'your', 'having', 'myself', 'after', 'will', 'all', 'himself', 'they', 'such', 'down', 'wouldn', 'how', 'between', 'yourselves', "needn't", 'wasn', 'which', "mustn't", 'should', "you've", 'them', "don't", 'it', 'then', 'y', 'ourselves', 'this', 'the', 'other', "haven't", "shouldn't", 'hers', 'but', "mightn't", 'me', 'won', 'now', 'she', 'been', 'not', 'while', 'o', 'these', 'you', 'a', 'just', 'being', 'hasn', 'shan', 'our', 'don', 'what', 'some', 'before', 't', 'has', 'have', 'over', 'further', 'ma', 've', 'of', 'm', 'ain', 'ours', 'themselves', "isn't", 'through', 'itself', 'were', 'had', 'in', 'each', 'couldn', 'on', 're', 'didn', 'those', 'needn', 'again', 'mustn', 'be', "you'll", "that'll", 'there', 'both', 'hadn', "won't", 'out', 'am', 'no', 'its', 'under', 'at', 'their', "aren't", 'same', 'does', 'who', 'doing', 'that', 'during', 'very', "it's", 'against', "couldn't", 'i', 'we', 'because', 'are', 'here', 'above', 'by', "she's", 'until', 'why', 'haven', "shan't", 'doesn', 'he', 'her', 'yours', 'into', 'nor', 'theirs', 'an', 'off', 'his', 'is', "you'd", "you're", 'if', 'do', 'below', 'my', 'mightn', 'from', "wouldn't", 'as', 'was', 'with', 'up', 'so', 'than', 'll', 'd', "hadn't", 'where', 's', 'whom', 'or', 'him', 'more', 'can', 'once', 'too', 'few', 'only', "weren't", 'own', 'to', 'about', 'aren', "hasn't", "wasn't", "didn't", 'any', 'weren', "doesn't", 'herself', 'and', 'yourself', 'when', 'most',
# 'shouldn', 'isn'}

# add some stop words
new_stop_words = {'', '5', "'d", '0', '9', '8', '2', 'also', 'first', 'one', '7', 'two', '6', '3', '1', "'m", '4', 'three'}
stop_words.update(new_stop_words)
# remove some stop words
remove_stop_words = set(['can', 'her', 'don', 'down', 'haven', 'own'])
stop_words = stop_words - remove_stop_words

def process(docs):
    if not isinstance(docs, list):
        doc = docs.lower().strip()
        doc = en_nlp(doc)
        doc_ = []
        for token in doc:
            token = token.lemma_
            if token not in stop_words:
                doc_.append(token)
        return " ".join(doc_)
    else:
        docs_ = []
        for doc in tqdm(docs):
            doc = doc.lower().strip()
            doc = en_nlp(doc)
            doc_ = []
            for token in doc:
                token = token.lemma_
                if token not in stop_words:
                    doc_.append(token)
            docs_.append(" ".join(doc_))
        return docs_

def load_json(fn):
    with open(fn, "r") as f:
        datas = json.load(f)
    return datas

if __name__ == "__main__":
    all_captions = load_json("./datas/captions/all_captions.json")
    processed_all_captions = {}
    for source in ["image", "video"]:
        processed_all_captions["{}_captions".format(source)] = {}

    if os.path.exists("./datas/captions/processed_all_captions.json"):
        processed_all_captions_ = load_json("./datas/captions/processed_all_captions.json")

    for kk, kv in processed_all_captions_.items():
        for k, v in kv.items():
            processed_all_captions[kk][k] = v

    for source in ["image", "video"]:
        sub_set = all_captions["{}_captions".format(source)]
        for k, captions in sub_set.items():
            if "{}_captions".format(source) in processed_all_captions and k in processed_all_captions["{}_captions".format(source)]:
                print("{} already processed".format(k))
                continue

            print("processing {} data...".format(k))
            captions = process(captions)
            processed_all_captions["{}_captions".format(source)][k] = captions
        
    with open("./datas/captions/processed_all_captions.json", "w") as f:
        json.dump(processed_all_captions, f)
    print("write all captions to ./datas/captions/processed_all_captions.json")