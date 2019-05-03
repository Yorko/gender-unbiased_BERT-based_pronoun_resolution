import numpy as np
import pandas as pd
import time
from contextlib import contextmanager
from tqdm import tqdm

tqdm.pandas()


# nice way to report running times
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def get_rank(token):
    i = 0
    next_token = token
    while next_token != next_token.head:
        i += 1
        next_token = next_token.head
    return i, next_token


def get_deptree_features(data):
    """
    Taken from public kernel:
    https://www.kaggle.com/negedng/extracting-features-from-spacy-dependency

    Computes positional information for A,B,P,
    measured with the dependency tree of sentences
    """

    features = []

    for i in range(0, len(data)):
        dataNext = data.loc[i]
        text = dataNext["Text"]

        doc = dataNext['spacy_nlp_doc']
        Aoff = dataNext["A-offset"]
        Boff = dataNext["B-offset"]
        Poff = dataNext["Pronoun-offset"]
        lth = len(text)

        for token in doc:
            if token.idx == Aoff:
                Atoken = token
            if token.idx == Boff:
                Btoken = token
            if token.idx == Poff:
                Ptoken = token

        Arank, Aroot = get_rank(Atoken)
        Brank, Broot = get_rank(Btoken)
        Prank, Proot = get_rank(Ptoken)

        sent_root = []
        for sent in doc.sents:
            sent_root.append(sent.root)

        sent_num = len(sent_root)
        for j in range(len(sent_root)):
            if Aroot == sent_root[j]:
                Atop = j
            if Broot == sent_root[j]:
                Btop = j
            if Proot == sent_root[j]:
                Ptop = j

        # normalized offsets
        fi = [Aoff / lth, Boff / lth, Poff / lth]

        # index of sentence where each word appears
        fi += [Atop / sent_num, Btop / sent_num, Ptop / sent_num]
 
        # distance from word to root of sentence
        fi += [Arank / 10, Brank / 10, Prank / 10]

        features.append(fi)
    return np.vstack(features)


def build_deptree_features(df):
    """
    Distances in dependency tree of sentences, as well as index of sentences

    :param df: pandas DataFrame with competition data
    :return: pandas DataFrame with 9 numeric features
    """

    with timer('Extracting deptree features'):
        deptree = get_deptree_features(df)
        columns = ["A_off", "B_off", "P_off",
                   "A_sent", "B_sent", "P_sent",
                   "A_rank", "B_rank", "P_rank"]
        deptree_df = pd.DataFrame(deptree, columns=columns)

    return deptree_df
