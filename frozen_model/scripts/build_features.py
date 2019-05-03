"""
Build train and test features.

Examples:
    # to check with a toy example
    python3 build_features.py -toy

    # build features for initial sets + precomputes e2e predictions and GAP heuristics
    python3 build_features.py -e2e -stanford_nlp

    # do the same for augmented data (without e2e and GAP for now)
    # here basically test_features_aug.tsv will be the same file as test_features.tsv
    # (we don't augment test data) - it's just to distinguish two files
    
    python3 build_features.py --train_file_name augmented_train.tsv.zip \
    --train_feat_file_name train_features_aug.tsv \
    --test_feat_file_name test_features_aug.tsv
"""
import pandas as pd
from pathlib import Path
import time
# needs to be installed with pip
import en_coref_md
from contextlib import contextmanager
from feat_group1_neuralcoref import get_neuralcoref_prediction
from feat_group2_subject_in_url import found_in_url
from feat_group3_syntactic import extract_syntactic
from feat_group4_position_and_freq import build_position_freq_features
from feat_group5_deptree import build_deptree_features
from feat_group6_named_ent import get_named_entities
from elmo_mlp_model import build_train_elmo
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-elmo', '--elmo', action="store_true", default=False,
                        help='Add precomputed elmo predictions.')
    parser.add_argument('-e2e', '--e2e', action="store_true", default=False,
                        help='Add precomputed e2e predictions.')
    parser.add_argument('-stanford_nlp', '--stanford_nlp', action="store_true", default=False,
                        help='Add precomputed stanford NLP predictions.')
    parser.add_argument('-gap', '--gap', action="store_true", default=False,
                        help='Add precomputed GAP heuristics from the original paper.')
    parser.add_argument('-toy', '--toy', action="store_true", default=False,
                        help='Whether to test with toy files. Prefix toy_.')
    parser.add_argument('-train', '--train_file_name', type=str, default='train.tsv')
    parser.add_argument('-test', '--test_file_name', type=str, default='test.tsv')
    parser.add_argument('-train_feat', '--train_feat_file_name', type=str, default='train_features.tsv',
                        help='Filename to output train features to')
    parser.add_argument('-test_feat', '--test_feat_file_name', type=str, default='test_features.tsv',
                        help='Filename to output test features to')
    parser.add_argument('-data', '--path_to_data', type=str, default='input')
    parser.add_argument('-features', '--path_to_features', type=str, default='features')
    return parser.parse_args()


# nice way to report running times
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


# it's better to have Spacy nlp model loaded rarely
# contains vocabulary, syntax, word embeddings for the english language
# you first need to install the model (pip install MODEL_URL)
# https://github.com/huggingface/neuralcoref
# then you need to install specific versions of cymem and spacy (see requirements.txt)
with timer('Loading Spacy model'):
    SPACY_MODEL = en_coref_md.load()


def build_features(df, spacy_model=SPACY_MODEL):
    """

    :param df: pandas DataFrame with competition data
    :param spacy_model: loaded Spacy model
    :return: pandas DataFrame with features
    """
    
    # first we add precomputed spacy docs for each sentence as a new Series
    df['spacy_nlp_doc'] = [spacy_model(df['Text'].iloc[i]) for i in range(len(df))]

    # Feature group 1  - spacy neural coref model prediction
    f1_neural_coref = get_neuralcoref_prediction(df)

    # Feature group 2 - whether A or B are found in URL
    f2_subj_in_url = found_in_url(df)

    # Feature group 3 - the syntactic role of A, B, Pronoun (subject, object, etc)
    f3_syntactic = extract_syntactic(df)

    # Feature group 4 - positional and frequency-based
    f4_pos_freq = build_position_freq_features(df)

    # Feature group 5 - dependency tree
    f5_deptree = build_deptree_features(df)

    # Feature group 6 - named entities
    f6_ent = get_named_entities(df)

    # Feature group 7 - to be added here
    pass

    # combine
    feat_df = pd.concat([f1_neural_coref, f2_subj_in_url,
                         f3_syntactic, f4_pos_freq,
                         f5_deptree, f6_ent], axis=1)

    return feat_df


def append_external_features(feat_df, ext_feat_file, sep='\t'):
    """

    :param feat_df: file with generated features
    :param ext_feat_file: path to external features
    :param sep: separator
    :return:
    """

    ext_feat_df = pd.read_csv(ext_feat_file, sep=sep, index_col=0)
    all_feat_df = pd.concat([feat_df, ext_feat_df], axis=1)

    return all_feat_df


def main(in_file, out_file, e2e_file, gap_file, stanford_file, elmo_file, args, sep='\t'):
    """
    Reads data, runs `build_features` and saves features to file.

    :param in_file: path to data
    :param out_file: path to dump features
    :param e2e_file: path to file with e2e features
    :param gap_file: path to file with GAP heuristics
    :param stanford_file: path to file with Stanford NLP predictions
    :param elmo_file: path to file with ELMo embeddings
    :param args: argparse args
    :param sep: separator
    :return:
    """

    if args.toy:
        in_file = in_file.parent / ('toy_' + in_file.name)        
        out_file = out_file.parent / ('toy_' + out_file.name) 
        e2e_file = e2e_file.parent / ('toy_' + e2e_file.name) 
        gap_file = gap_file.parent / ('toy_' + gap_file.name)
        stanford_file = stanford_file.parent / ('toy_' + stanford_file.name)

    df = pd.read_csv(in_file, sep=sep)
    feat_df = build_features(df)

    args.e2e = True
    args.gap = True
    args.stanford_nlp = True
    args.elmo = True # Leave this flag on for ELMo features

    if args.e2e:
        feat_df = append_external_features(feat_df, e2e_file, sep=sep)
    if args.gap:
        feat_df = append_external_features(feat_df, gap_file, sep=sep)
    if args.stanford_nlp:
        feat_df = append_external_features(feat_df, stanford_file, sep=sep)
    if args.elmo:
        feat_df = append_external_features(feat_df, elmo_file, sep=sep)

    feat_df.to_csv(out_file, sep=sep)


if __name__ == '__main__':

    args = parse_args()
    
    PATH_TO_DATA = Path(args.path_to_data)
    PATH_TO_FEATURES = Path(args.path_to_features)

    # This creates ELMo embeddings, trains a simple MLP model on them,
    # and outputs the predictions in `train_elmo_prob.tsv`, `test_elmo_prob.tsv`
    with timer('Getting ELMo features'):
        build_train_elmo()

    with timer('Featurizing train'):
        main(in_file=PATH_TO_DATA/args.train_file_name,
             out_file=PATH_TO_FEATURES/args.train_feat_file_name,
             e2e_file=PATH_TO_FEATURES/'train_e2e.tsv',
             gap_file=PATH_TO_FEATURES/'train_gap_heuristics.tsv',
             stanford_file=PATH_TO_FEATURES/'train_stanford.tsv',
             elmo_file=PATH_TO_FEATURES/'train_elmo_prob.tsv',
             args=args)
    with timer('Featurizing test'):
        main(in_file=PATH_TO_DATA/args.test_file_name,
             out_file=PATH_TO_FEATURES/args.test_feat_file_name,
             e2e_file=PATH_TO_FEATURES/'test_e2e.tsv',
             gap_file=PATH_TO_FEATURES/'test_gap_heuristics.tsv',
             stanford_file=PATH_TO_FEATURES/'test_stanford.tsv',
             elmo_file=PATH_TO_FEATURES/'test_elmo_prob.tsv',
             args=args)



