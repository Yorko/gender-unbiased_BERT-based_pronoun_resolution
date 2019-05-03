"""
en_core_web_md needs to be installed via pip
"""

import numpy as np
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


def decide_syntactic(row):
    """
    parse a row from the competition data, and find the syntactic role of A, B, P

    :param row: pandas Series holding a row of competition data with added Spacy nlp doc for the Text field
    :return: syntactic role of A, B, P
    """

    nlp_doc = row['spacy_nlp_doc']

    # if A or B span multiple tokens, Spacy may classify some of these as
    # compound or punctuation, which is not useful. when this happens, we
    # will skip to the next token in the span
    useless = ["compound", "punct"]

    a_offset, b_offset, p_offset = row.loc["A-offset"], row.loc["B-offset"], row.loc["Pronoun-offset"]
    synt_a, synt_b, synt_p = None, None, None

    for (j, token) in enumerate(nlp_doc):
        if token.idx == p_offset:
            # dep (stands for dependency) is spacy's way of saying syntactic role
            synt_p = "P-" + token.dep_

        if token.idx == a_offset:
            cnt = 0
            new_token = nlp_doc[j + cnt]

            # first token in the span of A may not have a role that's descriptive enough
            # skip until it's not useless
            while new_token.dep_ in useless:
                cnt += 1
                new_token = nlp_doc[j + cnt]
            synt_a = "A-" + new_token.dep_

        if token.idx == b_offset:
            cnt = 0
            new_token = nlp_doc[j + cnt]

            # same for B
            while new_token.dep_ in useless:
                cnt += 1
                new_token = nlp_doc[j + cnt]
            synt_b = "B-" + new_token.dep_

    return [synt_a, synt_b, synt_p]


def to_ohe(row, columns):
    """
    one-hot encoding for the 3 categorical variables in row,
    but only encode the most common categories

    :param row: pandas Series with 3 values, the syntactic role of A, B, P
    :param columns: the names of the most common values, we discard all others
    :return: numpy array with ohe of row
    """
    ohe = np.zeros(len(columns))
    col_names = list(row)
    for (index, column_name) in enumerate(columns):
        if column_name in col_names:
            ohe[index] = 1
    return ohe


def extract_syntactic(df):
    """
    For each of A,B,Pronoun, see whether its syntactic role is one of the
    most common (subject, direct object, attribute etc)

    :param df: pandas DataFrame with competition data
    :return: pandas DataFrame with 29 binary features, which are OHE of 3 categorical features
    """

    with timer('Extracting syntactic features'):
        # get a DataFrame with 3 categorical variables: syntactic role of A, B, P
        synt_categorical = df.progress_apply(lambda row: decide_syntactic(row), axis=1, result_type='expand')

        # the 10 most common values, for each of A, B, P
        columns = ["A-dobj", "A-poss", "A-nsubj", "A-pobj", "A-appos", "A-attr", "A-conj", "A-nsubjpass", "A-dative",
                   "A-nmod", "A-npadvmod", "B-dobj", "B-poss", "B-nsubj", "B-pobj", "B-appos", "B-attr", "B-conj",
                   "B-nsubjpass", "B-dative", "B-nmod", "B-npadvmod", "P-dobj", "P-poss", "P-nsubj", "P-pobj", "P-attr",
                   "P-nsubjpass", "P-dative"
                   ]

        # one-hot encode the previous results, discarding values that are not in columns list
        synt_ohe = synt_categorical.progress_apply(lambda row: to_ohe(row, columns), axis=1,
                                                   result_type='expand').astype(int)
        synt_ohe.columns = columns

    return synt_ohe
