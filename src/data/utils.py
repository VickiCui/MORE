import torch
from PIL import Image
import json
import jsonlines
import numpy as np

templates = {
    'AtLocation': "<head> is located at <tail>",
    'IsA': "<head> is a <tail>",
    'RelatedTo': "<head> is related to <tail>",
    'HasContext': "<head> is used in the context of <tail>",
    'PartOf': "<head> is part of <tail>",
    'HasA': "<head> has <tail>",
    'UsedFor': "<head> is used for <tail>",
    'CapableOf': "<head> can <tail>",
    'ReceivesAction': "<tail> can be done to <head>",
    'SimilarTo': "<head> is very like <tail>",
    'HasProperty': "<head> is usually <tail>",
    'Desires': "<head> wants <tail>",
    'Causes': "<head> causes <tail>",
    'Entails': "<head> entails <tail>",
    'MannerOf': "<head> is a way to <tail>",
    'MadeOf': "<head> is made of <tail>",
    'CreatedBy': "<head> is created by <tail>",
    'LocatedNear': "<head> is located near <tail>",
    'HasSubevent': "when you do <head> , you need to <tail>",
    'MotivatedByGoal': "<tail> is the reason of <head>",
    'SymbolOf': "<head> is the symbol of <tail>",
    'HasPrerequisite': "when you do <head> , you need to <tail>",
    'DefinedAs': "<head> is defined as <tail>",
}
def load_jsonl(fname):
    datas = []
    with open(fname, "r") as f:
        for item in jsonlines.Reader(f):
            datas.append(item)
            
    return datas

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
            
    return data

def get_image_input(process, file_names):
    image_inputs = []
    if isinstance(file_names, str):
        file_names = [file_names]
    for file_name in file_names:
        image = Image.open(file_name)
        image_input = process(image) # (3, 224, 224)
        image_inputs.append(image_input)
    image_inputs = torch.stack(image_inputs) # (n, 3, 224, 224)
    
    return image_inputs

def get_graph_input(tokenizer, graph_obj, paths):
    sents = []
    for path in paths:
        s,r,o = graph_obj.id2ent[int(path[0])], graph_obj.id2rel[int(path[1])], graph_obj.id2ent[int(path[2])]
        sent = templates[r].replace("<head>", s).replace("<tail>", o)
        sents.append(sent)

    # for path in paths:
    #     for rel in path["rels"]:
    #         sent = templates[rel].replace("<head>", path["head"]).replace("<tail>", path["tail"])
    #         sents.append(sent)

    graph_text = ". ".join(sents)
    graph_input_ids = tokenizer(graph_text, max_length=512, truncation=True)["input_ids"]

    return graph_input_ids

def get_text_input(tokenizer, text_input, feature_type="local"):
    if feature_type == "local":
        sent = " [SEP] ".join(text_input)
        input_ids = tokenizer(sent, max_length=512, truncation=True)["input_ids"]
    else:
        input_ids = tokenizer(text_input, max_length=512, padding=True, truncation=True, return_tensors='pt')["input_ids"]

    return input_ids

def _torch_collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    bsz = len(examples)
    # Tensorize if necessary.

    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    if len(examples[0].shape) == 2:
        # B, N, L
        dim = 3
        examples_ = []
        for example in examples:
            for instance in example:
                examples_.append(instance)
        examples = examples_
    else:
        dim = 2

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        result = torch.stack(examples, dim=0)
        if dim == 3:
            result = result.view(bsz, -1, result.size(-1))
            return result

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)

    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example

    if dim == 3:
        result = result.view(bsz, -1, max_length)
        
    return result