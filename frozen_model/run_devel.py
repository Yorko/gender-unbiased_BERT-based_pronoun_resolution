import os
import pandas as pd
import shutil
import time
from bert_utils import *
from bert_gap import BertGAP
from gap_model_triplet import GAPModelTriplet


def preprocess_embeddings():
    bert_gap = BertGAP(
        bert_model='models/uncased_L-24_H-1024_A-16',
        emb_size=1024,
        seq_len=64,
        n_layers=3,
        start_layer=4,
        do_lower_case=True,
        normalize_text=True)

    bert_gap.process_embeddings(
        input_file='gap-coreference/gap-test.tsv',
        sep='\t',
        output_file='input/emb_uncased_test.json',
        aug_id=0)

    bert_gap.process_embeddings(
        input_file='gap-coreference/gap-validation.tsv',
        sep='\t',
        output_file='input/emb_uncased_validation.json',
        aug_id=0)

    bert_gap.process_embeddings(
        input_file='gap-coreference/gap-development.tsv',
        sep='\t',
        output_file='input/emb_uncased_development.json',
        aug_id=0)


    bert_gap = BertGAP(
        bert_model='models/cased_L-24_H-1024_A-16',
        emb_size=1024,
        seq_len=64,
        n_layers=3,
        start_layer=4,
        do_lower_case=False,
        normalize_text=True)

    bert_gap.process_embeddings(
        input_file='gap-coreference/gap-test.tsv',
        sep='\t',
        output_file='input/emb_cased_test.json',
        aug_id=0)

    bert_gap.process_embeddings(
        input_file='gap-coreference/gap-validation.tsv',
        sep='\t',
        output_file='input/emb_cased_validation.json',
        aug_id=0)

    bert_gap.process_embeddings(
        input_file='gap-coreference/gap-development.tsv',
        sep='\t',
        output_file='input/emb_cased_development.json',
        aug_id=0)



def preprocess():

    # Create necessary dirs and downloads BERT and ELMo
    #os.system('sh preparation.sh')

    # Create test/train DFs
    train_dfs = []
    for filename in ['gap-coreference/gap-development.tsv',
                     'gap-coreference/gap-validation.tsv',
                     ]:
        df = pd.read_csv(filename, sep='\t')
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df.to_csv('input/train.tsv', sep='\t', index=False)
    shutil.copyfile('gap-coreference/gap-test.tsv', 'input/test.tsv')

    # Create features
    os.system('python3 scripts/build_features.py')

    # Create embeddings
    #preprocess_embeddings()


def train_model_triplet():
    model = GAPModelTriplet()
    embeddings = [
        {
            'train': [
                'input/emb_uncased_development.json',
                'input/emb_uncased_validation.json'
            ],
            'test': ['input/emb_uncased_test.json', ]
        },
        {
            'train': [
                'input/emb_cased_development.json',
                'input/emb_cased_validation.json'
            ],
            'test': ['input/emb_cased_test.json', ]
        },
    ]
    features = {
        'train': ['features/train_features.tsv', ],
        'train_ids': ['input/train.tsv', ],
        'test': ['features/test_features.tsv', ],
        'test_ids': ['input/test.tsv', ],
    }
    input_files = {
        'train': ['gap-coreference/gap-development.tsv',
                  'gap-coreference/gap-validation.tsv'],
        'test': ['gap-coreference/gap-test.tsv', ]
    }

    model.train_model(embeddings=embeddings, features=features, input_files=input_files)


def main():
    print("Started preprocessing at ", time.ctime())
    preprocess()
    print("Started training at ", time.ctime())
    train_model_triplet()


if __name__ == '__main__':
    main()

