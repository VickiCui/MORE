from datasets import load_metric
import random

def acc_metric_builder(tokenizer):
    def compute_ident_metrics(pred):
        """Utility to compute acc during training."""
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # All special tokens are removed.
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        correct = 0
        for (p, l) in zip(pred_str, label_str):
            if p.strip() == l.strip():
                correct += 1
        acc = correct / len(pred_str)

        idxs = random.sample(range(len(pred_str)), 10)
        for idx in idxs:
            print("label: {}".format(label_str[idx]))
            print("pred: {}".format(pred_str[idx]))

        return {
            "acc": round(acc * 100, 4),
        }

    return compute_ident_metrics

def span_metric_builder(tokenizer):
    def compute_span_metrics(pred):
        """Utility to compute ROUGE during training."""
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # All special tokens are removed.
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        iou = 0

        try:
            if label_str == "-1,-1":
                if pred_str == "-1,-1":
                    iou = 1
            else:
                s, e = pred_str.split(",")
                if s <= e:
                    s_gt, e_gt = label_str.split(",")

                    left = max(s, s_gt)
                    right = min(e, e_gt)

                    intersection = max(0, right-left)
                    union = (e - s) + (e_gt - s_gt) - intersection

                    union = max(union, 1)

                    iou = max(intersection/union, 0)
        except:
            pass

        idxs = random.sample(range(len(pred_str)), 10)
        for idx in idxs:
            print("label: {}".format(label_str[idx]))
            print("pred: {}".format(pred_str[idx]))

        return {
            "iou": round(iou, 4),
        }
    return compute_span_metrics



def rouge_metric_builder(tokenizer):
    rouge_scorer = load_metric("./src/metric/rouge.py")
    def compute_rouge_metrics(pred):
        """Utility to compute ROUGE during training."""
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # print("========")
        # print(pred_ids.shape)
        # print(pred_ids)

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
        rouge_results = rouge_scorer.compute(
            predictions=pred_str,
            references=label_str,
            rouge_types=["rouge2", "rougeL"],
            use_agregator=True,
            use_stemmer=False,
        )
        return {
            "rouge2": round(rouge_results['rouge2'].mid.fmeasure, 4),
            "rougeL": round(rouge_results['rougeL'].mid.fmeasure, 4),
        }
    return compute_rouge_metrics
