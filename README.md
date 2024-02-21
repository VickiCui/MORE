## MORE (Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning)

Code release for paper: [MORE:**M**ulti-m**O**dal **RE**trieval Augmented Generative Commonsense Reasoning](to_be_add). We propose a multi-modal retrieval augmented framework to assist LMs and MLMs in generating more sensible (commonsense-compliant) sentences.

## Install
```
conda create -n more python=3.8
conda activate more
pip install -r requirements.txt
pip install datas/en_core_web_sm-3.0.0-py3-none-any.whl
```

## Retrieved Data

You can download data crawled by us from [this link](to_be_add), or crawl by yourself:
```bash
cd ./datas/commongen
#image
python src/grounding_image/crawl_image.py
python src/grounding_image/remove_repeat_image.py

#text
python src/grounding_text/crawl_text.py --n_threads 8
```
In the downloaded data, the bing_images archive (174G in total) is split into parts of 5GB, merge data through:
```bash
cat bing_images.tar.gz.* > bing_images.tar.gz
tar -xzvf bing_images.tar.gz
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


## License
The code is licensed under the [MIT license](./LICENSE) and the crawled dataset is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.


