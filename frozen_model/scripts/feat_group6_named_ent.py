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


def get_named_entities(df):
    """
    Count the named entities that are neither A nor B.
    Hopefully this correlates with class "Neither".

    :param df: competition data with one extra field spacy_nlp_doc: precomputed nlp(text)
    :return:
    """

    named_df = pd.DataFrame(0, index=df.index, columns=["named_ent"])

    with timer('Extracting named entities'):
        for i in range(len(df)):
            doc = df.loc[i, "spacy_nlp_doc"]
            A = df.loc[i, "A"]
            B = df.loc[i, "B"]
            A_offset = df.loc[i, "A-offset"]
            B_offset = df.loc[i, "B-offset"]
            P_offset = df.loc[i, "Pronoun-offset"]

            # count persons that are not A or B
            # spacy's entities are spans, not tokens
            # e.g. "Cheryl Cassidy" is one entity
            ent_list = [ent for ent in doc.ents if (ent.label_ == "PERSON" and ent.text != A and ent.text != B)]
            named_df.loc[i, "named_ent"] = len(ent_list)

    return named_df
