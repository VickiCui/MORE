## MORE (Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning)

Code release for paper: [MORE:**M**ulti-m**O**dal **RE**trieval Augmented Generative Commonsense Reasoning](https://aclanthology.org/2024.findings-acl.69/) (ACL Findings 2024). We propose a multi-modal retrieval augmented framework to assist LMs and MLMs in generating more sensible (commonsense-compliant) sentences.

> **MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning** <br>
> Wanqing Cui, Keping Bi, Jiafeng Guo, Xueqi Cheng

[![](https://img.shields.io/badge/-code-green?style=flat-square&logo=github&labelColor=gray)](https://github.com/VickiCui/MORE)
[![](https://img.shields.io/badge/arXiv-2305.01928-b31b1b?style=flat-square)](https://arxiv.org/abs/2402.13625)

## Install
```
conda create -n more python=3.8
conda activate more
pip install -r requirements.txt
pip install datas/en_core_web_sm-3.0.0-py3-none-any.whl
```

## Retrieved Data

You can download data crawled by us from [this link](https://huggingface.co/datasets/VickiCui/MORE), or crawl by yourself:
```bash
cd ./datas/commongen
#image
python src/grounding_image/crawl_image.py
python src/grounding_image/remove_repeat_image.py

#text
python src/grounding_text/crawl_text.py --n_threads 8
```


## Get Features
Extract image and text features in advance and store them as .lmdb to speed up training. The following scripts will result in two folders, i.e., `blip2_image_feats.lmdb` and `blip2_text_feats.lmdb` located at `./datas/commongen/`

```bash
#image
python src/grounding_image/save_feat2lmdb_commongen.py
#text
python src/grounding_text/save_feat2lmdb_commongen.py
```

## Training
The bash for training:

```bash
bash scripts/train_more.sh
```
The outputs will be saved in `--output_dir ./res/more`


## Evaluation
Information about evaluation can be found at [CommonGen github link](https://github.com/INK-USC/CommonGen/tree/master/evaluation)

## Citation
If you find our work helpful, please consider cite as follows:

```
  @inproceedings{Cui2024MOREMR,
      title = "{MORE}: Multi-m{O}dal {RE}trieval Augmented Generative Commonsense Reasoning",
      author = "Cui, Wanqing  and
        Bi, Keping  and
        Guo, Jiafeng  and
        Cheng, Xueqi",
      booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
      month = aug,
      year = "2024",
      address = "Bangkok, Thailand and virtual meeting",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2024.findings-acl.69",
      doi = "10.18653/v1/2024.findings-acl.69",
      pages = "1178--1192",
}
```

## License
The code is licensed under the [MIT license](./LICENSE) and the crawled dataset is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.


