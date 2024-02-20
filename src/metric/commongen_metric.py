from datasets import load_metric
import random
import os
import numpy as np
import json
from src.eval.bleu.bleu import Bleu
from src.eval.meteor.meteor import Meteor
from src.eval.rouge.rouge import Rouge
from src.eval.cider.cider import Cider
from src.eval.spice.spice import Spice
import spacy
import sys
import codecs
import argparse

nlp = spacy.load("en_core_web_sm")
# nlp.pipeline = [('tagger', nlp.tagger)]

def tokenize(dict):
    for key in dict:
        new_sentence_list = []
        for sentence in dict[key]:
            a = ''
            for token in nlp((sentence)):
                a += token.text.lower()
                a += ' '
            if not a.endswith('. '):
                a += '. '
            new_sentence_list.append(a.rstrip())
        dict[key] = new_sentence_list
    return dict

def commongen_metric_builder(tokenizer, key_file, gts_file):
    with codecs.open(key_file, encoding='utf-8') as f:
        key_lines = f.readlines()
    with codecs.open(gts_file, encoding='utf-8') as f:
        gts_lines = f.readlines()

    with open(key_file, "r") as f:
        concept_sets = [item.split() for item in f.read().split("\n")]

    def coverage_score(preds, concept_sets):
        if len(preds) != len(concept_sets):
            concept_sets_ = []
            for c in concept_sets:
                if c not in concept_sets_:
                    concept_sets_.append(c)
            concept_sets = concept_sets_
            
        covs = []
        for p, cs in zip(preds,concept_sets):
            cs = set(cs)
            lemmas = set()
            for token in nlp((p)):
                lemmas.add(token.lemma_)
            cov = float(len(lemmas&cs))/float(len(cs))
            covs.append(cov)
        return float(sum(covs))/float(len(covs))

    def coverage(preds):
        print("computing Coverage score...")
        c_score = coverage_score(preds, concept_sets)
        return c_score

    def evaluator(gts, res):
        eval = {}
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        # Todo: use Spacy for tokenization
        gts = tokenize(gts)
        res = tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Bleu(4), ["Bleu_4"]),
            # (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    eval[m] = sc
                    # print("%s: %0.3f" % (m, sc))
            else:
                eval[method] = score
                # print("%s: %0.3f" % (method, score))

        return eval

    def compute_metrics(pred):
        """Utility to compute ROUGE during training."""
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # All special tokens are removed.
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        # for i, pred_id in enumerate(pred_ids):
        #     try:
        #         tokenizer.batch_decode([pred_id], skip_special_tokens=True)
        #     except:
        #         print("pred_id", i, pred_id)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        # for i, labels_id in enumerate(labels_ids):
        #     try:
        #         tokenizer.batch_decode([labels_id], skip_special_tokens=True)
        #     except:
        #        print("labels_id", i, labels_id)

        # print("========")
        # print(labels_ids.shape)
        # print(labels_ids)

        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        idxs = random.sample(range(len(pred_str)), 10)
        for idx in idxs:
            print("========== {} ==========".format(idx))
            print("label: {}".format(label_str[idx]))
            print("pred: {}".format(pred_str[idx]))

        # Compute the metric.
        gts = {}
        res = {}
        i = -1
        for key_line, gts_line in zip(key_lines, gts_lines):
            key = '#'.join(key_line.rstrip('\n').split(' '))
            if key not in gts:
                i += 1
                if i >= len(pred_str):
                    break
                gts[key] = []
                gts[key].append(gts_line.rstrip('\n'))
                res[key] = []
                res[key].append(pred_str[i].rstrip('\n'))
            else:
                gts[key].append(gts_line.rstrip('\n'))
        res = evaluator(gts, res)

        score = coverage(pred_str)
        # print("Coverage: %0.3f" % score)
        res["Coverage"] = score
        del res['Bleu_1']
        del res['Bleu_2']
        del res['Bleu_3']

        avg = []
        for k, v in res.items():
            if k == "Coverage":
                continue

            if k != "CIDEr":
                avg.append(v*100)
            else:
                avg.append(v*10)
        res["average"] = np.mean(avg)
        return res
        # rouge_results = rouge_scorer.compute(
        #     predictions=pred_str,
        #     references=label_str,
        #     rouge_types=["rouge2", "rougeL"],
        #     use_agregator=True,
        #     use_stemmer=False,
        # )
        # return {
        #     "rouge2": round(rouge_results['rouge2'].mid.fmeasure, 4),
        #     "rougeL": round(rouge_results['rougeL'].mid.fmeasure, 4),
        # }
    return compute_metrics