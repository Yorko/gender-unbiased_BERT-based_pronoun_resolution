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


def sentence_positional_features(row):
    """
    This creates a set of binary variables indicating whether all A, B and pronoun are in the same sentence
    or pronoun is placed in the next one. Or A and pronoun are in one sentence, but B - is in the next one. etc.

    :param row: row in the training dataframe
    :return: a list of binary variables
    """
    min_offset = min([row['A-offset'], row['B-offset'], row['Pronoun-offset']])
    max_offset = max([row['A-offset'], row['B-offset'], row['Pronoun-offset']])

    text_slice = row['Text'][min_offset:max_offset]

    new_a_offset = row['A-offset'] - min_offset
    new_b_offset = row['B-offset'] - min_offset
    new_pron_offset = row['Pronoun-offset'] - min_offset

    all_in_one_sent = int('.' not in text_slice)

    pronoun_in_next_sent, b_in_next_sent = 0, 0
    pronoun_in_prev_sent, a_in_prev_sent = 0, 0

    if (('.' in text_slice[max(new_a_offset, new_b_offset):new_pron_offset])
        or (';' in text_slice[max(new_a_offset, new_b_offset):new_pron_offset])) \
            and (new_a_offset < new_pron_offset) and (new_b_offset < new_pron_offset):
        pronoun_in_next_sent = 1

    if (('.' in text_slice[max(new_a_offset, new_pron_offset):new_b_offset])
        or (';' in text_slice[max(new_a_offset, new_pron_offset):new_b_offset])) \
            and (new_a_offset < new_b_offset) and (new_pron_offset < new_b_offset):
        b_in_next_sent = 1

    if (('.' in text_slice[new_pron_offset:min(new_a_offset, new_b_offset)])
        or (';' in text_slice[new_pron_offset:min(new_a_offset, new_b_offset)])) \
            and (new_pron_offset < new_a_offset) and (new_pron_offset < new_b_offset):
        pronoun_in_prev_sent = 1

    if (('.' in text_slice[new_a_offset:min(new_pron_offset, new_b_offset)]) or
        (';' in text_slice[new_a_offset:min(new_pron_offset, new_b_offset)])) \
            and (new_a_offset < new_b_offset) and (new_a_offset < new_pron_offset):
        a_in_prev_sent = 1

    return [all_in_one_sent, pronoun_in_next_sent, b_in_next_sent,
            pronoun_in_prev_sent, a_in_prev_sent]


def position_features(df):
    """

    :param df: pandas DataFrame with competition data
    :return: pandas DataFrame with new features
    """

    pred_df = pd.DataFrame(index=df.index)

    pred_df['A_then_pronoun'] = (df['A-offset'] < df['Pronoun-offset']).astype('uint8')
    pred_df['B_then_pronoun'] = (df['B-offset'] < df['Pronoun-offset']).astype('uint8')
    pred_df['pronoun_first'] = ((df['Pronoun-offset'] < df['A-offset']) &
                                (df['Pronoun-offset'] < df['B-offset'])).astype('uint8')
    pred_df['pronoun_last'] = ((df['Pronoun-offset'] > df['A-offset']) &
                               (df['Pronoun-offset'] > df['B-offset'])).astype('uint8')
    pred_df['A-dist'] = (df['Pronoun-offset'] - df['A-offset']).abs()
    pred_df['B-dist'] = (df['Pronoun-offset'] - df['B-offset']).abs()

    return pred_df


def build_position_freq_features(df):
    """

    Counts numbers of A and B in Text, as well as

    :param df: pandas DataFrame with competition data
    :return: pandas DataFrame with new features
    """

    pred_df = pd.DataFrame(index=df.index)

    with timer('Frequencies of A and B in text'):
        pred_df['freq_A'] = df.progress_apply(lambda row: row['Text'].count(row['A']), axis=1)
        pred_df['freq_B'] = df.progress_apply(lambda row: row['Text'].count(row['B']), axis=1)

    with timer('Positional features'):
        pred_df = pd.concat([pred_df, position_features(df)], axis=1)

        new_feats = ['all_in_one_sent', 'pronoun_in_next_sent', 'b_in_next_sent',
                     'pronoun_in_prev_sent', 'a_in_prev_sent']

        pred_df = pd.concat([pred_df, pd.DataFrame([sentence_positional_features(row)
                                                    for i, row in df.iterrows()],
                                                   columns=new_feats)], axis=1)

    return pred_df
