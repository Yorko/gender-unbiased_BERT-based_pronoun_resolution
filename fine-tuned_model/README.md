# BERT fine tuning for GAP data
Fine tuning of BERT with parsing of GAP data for [this kaggle competition](https://www.kaggle.com/c/gendered-pronoun-resolution)

This repository is based on [BERT](https://github.com/google-research/bert)

The actual BERT model should be downloaded separately. On a GPU you will have more parameter options with [BERT base](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) but [BERT large](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip) run on a TPU will give better results. The classifier in this repo can be used on GPU or TPU.
