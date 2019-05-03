"""
Get Spacy Neuralcoref prediction https://github.com/huggingface/neuralcoref
en_coref_md needs to be installed via pip.


"""

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


def get_coref(row):
    """

    Get Spacy Neuralcoref prediction https://github.com/huggingface/neuralcoref
    Taken from Public kernels.

    :param row: pandas Series, assume that we added Spacy NLP doc (nlp(text)) as one of the fields
    :return: coreference, string
    """
    coref = None

    nlp_doc = row['spacy_nlp_doc']
    # dunno if more direct way to get token from text offset
    for tok in nlp_doc.doc:
        if tok.idx == row['Pronoun-offset']:
            # model limitation that sometimes there are no coref clusters for the token?
            # also, sometimes the coref clusters will just be something like:
            # He: his, him, his
            # So there is no proper name to map back to?
            try:
                if len(tok._.coref_clusters) > 0:
                    coref = tok._.coref_clusters[0][0].text
            except:
                # for some, get the following exception just checking len(tok._.coref_clusters)
                # *** TypeError: 'NoneType' object is not iterable
                pass
            break

    if coref:
        coref = coref.lower()
        # sometimes the coref is I think meant to be the same as A or B, but
        # it is either a substring or superstring of A or B
        a_noun = row['A'].lower()
        b_noun = row['B'].lower()
        if coref in a_noun or a_noun in coref:
            coref = a_noun
        elif coref in b_noun or b_noun in coref:
            coref = b_noun

    return coref


def get_neuralcoref_prediction(df):
    """
    Apply neuralcoref prediction to the whole DataFrame.

    :param df: pandas DataFrame with competition data
    :return: pandas DataFrame with 2 binary features 'neuralcoref_A'
    """

    with timer('Neuralcoref prediction'):
        pred_df = pd.DataFrame(index=df.index)
        coref_prediction = df.progress_apply(lambda row: get_coref(row), axis=1)
        pred_df['neuralcoref_A'] = (coref_prediction == df['A'].str.lower()).astype('int')
        pred_df['neuralcoref_B'] = (coref_prediction == df['B'].str.lower()).astype('int')
    return pred_df




