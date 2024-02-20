from transformers import HfArgumentParser, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoConfig, AutoTokenizer
from datasets import load_metric 
import datasets
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.deepspeed import is_deepspeed_zero3_enabled

import logging
import os
import sys
import pandas as pd
import torch
from dataclasses import dataclass, field
import random

from model.ke_t5 import KnowledgeEnhancedT5
from data.aipoc import AipocDataset, DataCollatorAipoc
from data.commongen import CommongenDataset, DataCollatorCommongen
from data.init_graph import ConceptNet
from metric.metric_builder import ident_metric_builder, span_metric_builder, rouge_metric_builder
from metric.commongen_metric import commongen_metric_builder

metric_map = {
    "nonsense": ident_metric_builder,
    "span": span_metric_builder,
    "org_sent": rouge_metric_builder,
    "": rouge_metric_builder,
}

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
if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.optim import OSS

from transformers.file_utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
from transformers.trainer_pt_utils import get_parameter_names

class MySeq2SeqTrainer(Seq2SeqTrainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.former_learning_rate is None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "language_model" not in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.former_learning_rate,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "language_model" not in n)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.former_learning_rate,
                    },
                    
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "language_model" in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "language_model" in n)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        # for param_group in self.optimizer.param_groups:# 
        #     print('学习率: ',param_group['lr'])
        # exit()

        return self.optimizer

@dataclass
class MyArguments():
    """
    Data Arguments
    """
    data_dir: str = field(
        default="datas/commongen",
        metadata={"help": "The input data dir. Should contain the .json files (or other data files) for the task."}
    )

    label_field: str = field(
        default="",
        metadata={"help": "used to assign tasks: nonsense: identification; span: positioning; org_sent: correction."}
    )

    task_prompt: str = field(
        default=None,
        metadata={"help": "Prompt for task, e.g. Is the sentence against commonsense: "}
    )

    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    ### image ###
    image_model_path: str = field(
        default="./pre_trained_lm/clip/ViT-B-16.pt", 
        metadata={"help": "Path to pretrained clip model or model identifier from huggingface.co/models"}
    )

    use_image: bool = field(
        default=False,
        metadata={"help": "If use image as knowledge argumentation"}
    )

    rand_image: bool = field(
        default=False,
        metadata={"help": "Ablation study for the effect of parameters"}
    )

    image_grounding_path: str = field(
        default=None, 
        metadata={"help": "Path to grounding image information. Needed when use_image is True"}
    )

    whiteboard_image: bool = field(
        default=False,
        metadata={"help": "Useless!"}
    )

    image_input_path: str = field(
        default=None, 
        metadata={"help": "Lmdb path of the grounding image input. Not necessary. For accelerate training"}
    )

    image_feat_path: str = field(
        default=None, 
        metadata={"help": "Lmdb path of the grounding image input. Not necessary. For accelerate training"}
    )

    image_num: int = field(
        default=5
    )

    ### graph ###
    use_graph: bool = field(
        default=False,
        metadata={"help": "If use kb as knowledge argumentation"}
    )

    graph_grounding_path: str = field(
        default=None, 
        metadata={"help": "Never use for gcn. Path to grounding graph information. Needed when use_graph is True"}
    )

    whiteboard_graph: bool = field(
        default=False,
        metadata={"help": "Useless!"}
    )

    graph_input_path: str = field(
        default=None, 
        metadata={"help": "Lmdb path of the grounding graph input. Not necessary. For accelerate training"}
    )

    graph_model_path: str = field(
        default=None, 
        metadata={"help": "GCN path of the graph encoder."}
    )

    gcn_layer: int = field(
        default=1,
        metadata={"help": "Model parameter of GCN."}
    )

    graph_num: int = field(
        default=0,
         metadata={"help": "Neighbor number for each concept."}
    )

    init_dim: int = field(
        default=300,
        metadata={"help": "Model parameter of GCN."}
    )

    gcn_dim: int = field(
        default=200,
        metadata={"help": "Model parameter of GCN."}
    )

    embed_dim: int = field(
        default=100,
        metadata={"help": "Model parameter of GCN."}
    )

    init_graph_emb_path: str = field(
        default=None,
    )

    ### text ###
    text_model_path: str = field(
        default="./pre_trained_lm/bert-base-uncased", 
        metadata={"help": "Path to text encoder"}
    )

    use_text: bool = field(
        default=False,
        metadata={"help": "If use kb as knowledge argumentation"}
    )

    text_grounding_path: str = field(
        default=None, 
        metadata={"help": "Path to grounding graph information. Needed when use_graph is True"}
    )

    text_num: int = field(
        default=3
    )

    rand_text: bool = field(
        default=False,
        metadata={"help": "Ablation study for the effect of parameters"}
    )

    num_query_token: int = field(
        default=32
    )

    debug_mode: bool = field(
        default=False,
    )

    random_ignore: float = field(
        default=0,
        metadata={"help": "Random mask probability of each modality. -1 for never mask"}
    )

    random_shuffle: bool = field(
        default=False,
        metadata={"help": "Whether shuffle the order of concept when training"}
    )

    """
    Model Arguments
    """

    freeze_image_encoder: bool = field(
        default=True,
    )

    freeze_text_encoder: bool = field(
        default=True,
    )

    freeze_graph_encoder: bool = field(
        default=True,
    )

    model_name_or_path: str = field(
        default="./pre_trained_lm/t5-base", 
        metadata={"help": "Path to pretrained vit model or model identifier from huggingface.co/models"}
    )

    integrator_name_or_path: str = field(
        default="./pre_trained_lm/bert-base-uncased", 
        metadata={"help": "Path to pretrained vit model or model identifier from huggingface.co/models"}
    )

    resume_checkpoint: str = field(
        default=None, 
        metadata={"help": "Path to pretrained vit model or model identifier from huggingface.co/models"}
    )

    freeze_lm: bool = field(
        default=False,
    )

    image_qformer_text_input: bool = field(
        default=False,
        metadata={"help": "If use kb as knowledge argumentation"}
    )

    text_qformer_text_input: bool = field(
        default=False,
        metadata={"help": "If use kb as knowledge argumentation"}
    )

    graph_qformer_text_input: bool = field(
        default=False,
        metadata={"help": "If use kb as knowledge argumentation"}
    )

    former_learning_rate: float = field(
        default=None
    )

    use_cl: float = field(
        default=0
    )

    use_score: float = field(
        default=0
    )
    
    force_top1: bool = field(
        default=False,
    )

    mask_input: float = field(
        default=0,
        metadata={"help": "Random mask probability of each modality. -1 for never mask"}
    )


def main():
    parser = HfArgumentParser((MyArguments, Seq2SeqTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    setattr(training_args, "former_learning_rate", args.former_learning_rate)

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

    # for k, v in vars(tuning_args).items():
    #     if not hasattr(config, k) or k in ["full_finetune"]:
    #         setattr(config, k, v)
    # print(config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path
    )

    integrator_tokenizer = AutoTokenizer.from_pretrained(
        "./pre_trained_lm/bert-base-uncased"
    )

    graph_object = None
    graph_pad_idx = None
    if args.use_graph:
        graph_object = ConceptNet()
        graph_pad_idx = graph_object.pad_idx

    model = KnowledgeEnhancedT5(
        config=config,
        args=args,

        freeze_image_encoder=args.freeze_image_encoder,
        num_image_query_token=args.num_query_token,

        freeze_graph_encoder=args.freeze_graph_encoder,
        num_graph_query_token=args.num_query_token,
        graph_object=graph_object,

        text_model_path=args.text_model_path,
        freeze_text_encoder=args.freeze_text_encoder,
        num_text_query_token=args.num_query_token,

        integrator_path=args.integrator_name_or_path,
        t5_model=args.model_name_or_path,
        max_txt_len=256,
        resume_checkpoint=args.resume_checkpoint,
    )

    # model.resize_token_embeddings(len(tokenizer))
    eval_dataset = CommongenDataset(args, tokenizer, integrator_tokenizer, "val", graph_object)
    test_dataset = CommongenDataset(args, tokenizer, integrator_tokenizer, "test", graph_object)
    data_collator = DataCollatorCommongen(tokenizer, integrator_tokenizer, graph_pad_idx)

    # from torch.utils.data import DataLoader, SequentialSampler
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)
    # for i, batch in enumerate(train_dataloader):
    #     print(batch)
    #     if i >= 3:
    #         exit()

    # logger.info("*** Evaluate ***")
    # metric_fn = commongen_metric_builder(tokenizer, "datas/commongen/commongen.dev.src_alpha.txt", "datas/commongen/commongen.dev.tgt.txt")

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=metric_fn if training_args.predict_with_generate else None,
    # )

    # eval_results = trainer.predict(eval_dataset, metric_key_prefix="eval")
    # metrics = eval_results.metrics
    # metrics["eval_samples"] = len(eval_dataset)
    # trainer.save_metrics("eval", metrics)

    # if trainer.is_world_process_zero():
    #     if training_args.predict_with_generate:
    #         predictions = tokenizer.batch_decode(
    #             eval_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #         )
    #         predictions = [pred.strip() for pred in predictions]
    #         output_prediction_file = os.path.join(training_args.output_dir, "eval_predictions.txt")
    #         with open(output_prediction_file, "w") as writer:
    #             writer.write("\n".join(predictions))

    
    metric_fn = commongen_metric_builder(tokenizer, "datas/commongen/commongen.test.src_alpha.txt", "datas/commongen/commongen.test.tgt.txt")

    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metric_fn if training_args.predict_with_generate else None,
    )

       
    predict_results = trainer.predict(
        test_dataset, metric_key_prefix="test",
    )
    metrics = predict_results.metrics
    metrics["predict_samples"] = len(test_dataset)
    trainer.save_metrics("predict", metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "test_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))

if __name__ == "__main__":
    main()
