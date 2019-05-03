import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

import os
os.environ["CORENLP_HOME"] = '/home/user/StanfordTemp/stanford-corenlp-full-2018-10-05'
from stanfordnlp.server import CoreNLPClient

train = pd.read_csv('../input/train.tsv', index_col=0, sep='\t').reset_index(drop=True)
test = pd.read_csv('../input/test.tsv', index_col=0, sep='\t').reset_index(drop=True)

print("start Stanford NLP")
# set up the client
with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner','parse','depparse','coref'], timeout=90000, memory='16G') as client:

    print("Stanford NLP created")

    def predict_stanford(df, filename) :

        df['Pronoun-offset2'] = df['Pronoun-offset'] + df['Pronoun'].map(len)
        df['A-offset2'] = df['A-offset'] + df['A'].map(len)
        df['B-offset2'] = df['B-offset'] + df['B'].map(len)

        df['section_min'] = df[['Pronoun-offset', 'A-offset', 'B-offset']].min(axis=1)
        df['section_max'] = df[['Pronoun-offset2', 'A-offset2', 'B-offset2']].max(axis=1)

        df['A-dist'] = (df['Pronoun-offset'] - df['A-offset']).abs()
        df['B-dist'] = (df['Pronoun-offset'] - df['B-offset']).abs()


        df['pred-stanford'] = df.progress_apply(lambda row : stanford_predict_row(row), axis=1)
        df["A-stanford"] = df['pred-stanford'].apply(lambda x : x[0])
        df["B-stanford"] = df['pred-stanford'].apply(lambda x : x[1])

        df[['A-stanford', 'B-stanford']].to_csv(filename, sep='\t')
        return df

    def stanford_predict_row(row) :
        text = row["Text"]
        l = row["Text"].rfind(".", 0, row["section_min"])
        r = row["Text"].find(".", row["section_max"])
        text = text[l+1:r]

        ann = client.annotate(text)

        # for mention in ann.mentionsForCoref :
        #     print("cluster {}({}) id = {}, num = {}, sent = {} pos = ({}, {}) TOKEN = {}".format(mention.corefClusterID, mention.goldCorefClusterID,
        #                                             mention.mentionID, mention.mentionNum, mention.sentNum, mention.startIndex, mention.endIndex,
        #                                 " ".join([token.word for token in ann.sentence[mention.sentNum].token[mention.startIndex : mention.endIndex]])))


        unique_clusters = set([mention.corefClusterID for mention in ann.mentionsForCoref])

        cluster_mentions = [[" ".join([token.word.replace(" \'s", "") for token in ann.sentence[mention.sentNum].token[mention.startIndex : mention.endIndex]])
                             for mention in ann.mentionsForCoref if mention.corefClusterID == cluster ] for cluster in unique_clusters ]

        # clusters = [mention.corefClusterID for mention in ann.mentionsForCoref]
        # mentionIDs = [mention.mentionID for mention in ann.mentionsForCoref]
        # mentionNums = [mention.mentionNum for mention in ann.mentionsForCoref]
        # mentionSentNums = [mention.sentNum for mention in ann.mentionsForCoref]
        # mentions = [" ".join([token.word.replace(" \'s", "") for token in ann.sentence[mention.sentNum].token[mention.startIndex : mention.endIndex]])
        #             for mention in ann.mentionsForCoref]

        pronounClusters = [cluster for cluster in cluster_mentions if row["Pronoun"] in cluster]
        count_A = len([cluster for cluster in pronounClusters if row["A"] in cluster])
        count_B = len([cluster for cluster in pronounClusters if row["B"] in cluster])

        return (1 if count_A > 0 else 0, 1 if count_B > 0 else 0)

    stanford_test = predict_stanford(test, "../features/test_stanford.tsv")
    stanford_train = predict_stanford(train, "../features/train_stanford.tsv")