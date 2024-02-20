from transformers import HfArgumentParser, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoConfig, AutoTokenizer
from datasets import load_metric 
import datasets
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import os
import sys
import pandas as pd
import torch
from dataclasses import dataclass, field
import random

from model.baseline_prepend import PrepandlKnowledgeEnhancedT5
from data.commongen_prepend import CommongenDataset, DataCollatorCommongen
from metric.commongen_metric import commongen_metric_builder

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

os.environ["WANDB_DISABLED"] = "true"

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import ShardedDDPOption
from transformers.integrations import is_fairscale_available
from transformers.dependency_versions_check import dep_version_check
if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.optim import OSS

from transformers.file_utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
from transformers.trainer_pt_utils import get_parameter_names

model_map = {
    "PrepandlKnowledgeEnhancedT5": PrepandlKnowledgeEnhancedT5,
}

@dataclass
class MyArguments():
    """
    Data Arguments
    """
    data_dir: str = field(
        default="datas/commongen",
        metadata={"help": "The input data dir. Should contain the .json files (or other data files) for the task."}
    )

    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    text_num: int = field(
        default=0
    )

    debug_mode: bool = field(
        default=False,
    )


    """
    Model Arguments
    """
    model_class: str = field(
        default="DimGatedAttnKnowledgeEnhancedT5"
    )

    model_name_or_path: str = field(
        default="./pre_trained_lm/t5-base", 
        metadata={"help": "Path to pretrained vit model or model identifier from huggingface.co/models"}
    )

    resume_checkpoint: str = field(
        default=None, 
        metadata={"help": "Path to pretrained vit model or model identifier from huggingface.co/models"}
    )

    freeze_lm: bool = field(
        default=True,
    )

    prompt_length: int = field(
        default=32
    )


def main():
    parser = HfArgumentParser((MyArguments, Seq2SeqTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    
    print(args)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path
    )


    model_class = model_map[args.model_class]

    model = model_class(
        config=config,
        args=args,
        prompt_length=args.prompt_length,
        t5_model=args.model_name_or_path,
        resume_checkpoint=args.resume_checkpoint,
    )

    # model.resize_token_embeddings(len(tokenizer))
    train_dataset = CommongenDataset(args, tokenizer, split="train")
    eval_dataset = CommongenDataset(args, tokenizer, split="val")
    data_collator = DataCollatorCommongen(tokenizer, None)

    metric_fn = commongen_metric_builder(tokenizer, "datas/commongen/commongen.dev.src_alpha.txt", "datas/commongen/commongen.dev.tgt.txt")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metric_fn if training_args.predict_with_generate else None,
    )

    if training_args.do_train:
        logger.info("*** Train ***")
        trainer.train(training_args.resume_from_checkpoint)
        trainer.save_model()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_results = trainer.predict(
            eval_dataset, metric_key_prefix="eval",
        )
        metrics = eval_results.metrics
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    eval_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "eval_generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    if training_args.do_predict:
        metric_fn = commongen_metric_builder(tokenizer, "datas/commongen/commongen.test.src_alpha.txt", "datas/commongen/commongen.test.tgt.txt")

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=metric_fn if training_args.predict_with_generate else None,
        )

        test_dataset = CommongenDataset(args, tokenizer, split="test")

        predict_results = trainer.predict(
            test_dataset, metric_key_prefix="test",
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(test_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "test_generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

if __name__ == "__main__":
    main()
