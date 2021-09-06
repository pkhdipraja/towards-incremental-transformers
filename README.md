## Towards Incremental Transformers: An Empirical Analysis of Transformer Models for Incremental NLU

### Setup
* Install python3 requirements: `pip install -r requirements.txt`
* Initialize GloVe as follows:
```bash
$ wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.3.0/en_vectors_web_lg-2.3.0.tar.gz -O en_vectors_web_lg-2.3.0.tar.gz
$ pip install en_vectors_web_lg-2.3.0.tar.gz
```

### Training
You should first create a model configuration file under `configs/` (see the provided sample). The following script will run the training:
```bash
$ python3 main.py --RUN train --MODEL_CONFIG <model_config> --DATASET <dataset>
```

with checkpoint saved under `ckpts/<dataset>/` and log under `results/log/`

Important parameters:
1. `--VERSION str`, to assign a name for the model.
2. `--GPU str`, to train the model on specified GPU. For multi-GPU training, use e.g. `--GPU '0, 1, 2, ...'`.
3. `--SEED int`, set seed for this experiment.
4. `--RESUME True`, start training with saved checkpoint. You should assign checkpoint version `--CKPT_V str` and resumed epoch `--CKPT_E int`.
5. `--NW int`, to accelerate data loading speed.
6. `--DATA_ROOT_PATH str`, to set path to your dataset.

To check all possible parameters, use `--help`


### Testing
You can evaluate on validation or test set using `--RUN {val, test}`. For example:
```bash
$ python3 main.py --RUN test --MODEL_CONFIG <model_config> --DATASET <dataset> --CKPT_V <model_version> --CKPT_E <model_epoch>
```
or with absolute path:
```bash
$ python3 main.py --RUN test --MODEL_CONFIG <model_config> --DATASET <dataset> --CKPT_PATH <path_to_checkpoint>.ckpt
```

To obtain incremental evaluation on the test set, use the flag `--INCR_EVAL`

### Data

We do not upload the original datasets as some of them needs license agreements. The preprocessing steps are described in the paper.

As it is, the code can run experiments on:

* Chunk, [CoNLL 2000](https://www.clips.uantwerpen.be/conll2000/chunking/) (chunk)
* Named Entity Recognition, [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19), WSJ (ner-nw-wsj)
* PoS Tagging, [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19), WSJ (pos-nw-wsj)
* Slot filling and intent detection, [ATIS](https://aclanthology.org/H90-1021/) (atis-slot & atis-intent)
* Slot filling and intent detection, [SNIPS](https://github.com/snipsco/nlu-benchmark) (snips-slot & snips-intent)
* Sentiment classification, [Pros/Cons](http://www.cs.uic.edu/~liub/FBS/pros-cons.rar) (proscons)
* Sentiment classification, [Positive/Negative](http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences) (sent-negpos)

Data has to be split into three files (`data/train/train.<task>`, `data/valid/valid.<task>` and `data/test/test.<task>`) as in `/configs/path_config.yml`, all of them following the format:

* Sequence tagging:
```yml
token \t label \n token \t label \n
```
with an extra \n between sequences.

* Sequence classification:
```yml
<LABEL>: atis_airfare \n token \n token \n
```
with an extra \n between sequences.

If this repository is helpful for your research, we would really appreciate if you could cite the paper:

```
@inproceedings{kahardipraja21,
    title = "Towards Incremental Transformers: An Empirical Analysis of Transformer Models for Incremental NLU",
    author = "Kahardipraja, Patrick  and
      Madureira, Brielen  and
      Schlangen, David",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "[To Appear]",
    pages = "[To Appear]",
}
```

