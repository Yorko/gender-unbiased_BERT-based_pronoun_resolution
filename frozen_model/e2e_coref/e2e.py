import sys
sys.path.append('../../Pronoun/e2e_coref')

import numpy as np
import pandas as pd

from tqdm import tqdm
tqdm.pandas()

import tensorflow as tf
import coref_model as cm
import util

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

def create_example(text):
    raw_sentences = sent_tokenize(text)
    sentences = [word_tokenize(s) for s in raw_sentences]
    speakers = [["" for _ in sentence] for sentence in sentences]
    return {
        "doc_key": "nw",
        "clusters": [],
        "sentences": sentences,
        "speakers": speakers,
    }

def print_predictions(example):
    words = util.flatten(example["sentences"])
    for cluster in example["predicted_clusters"]:
        print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))

def make_predictions(text, model):
    example = create_example(text)
    tensorized_example = model.tensorize_example(example, is_training=False)
    feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
    _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

    predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

    example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
    example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
    example["head_scores"] = head_scores.tolist()
    return example

def get_antecedents(text, model):
    example = create_example(text)
    tensorized_example = model.tensorize_example(example, is_training=False)
    feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
    _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

    predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

    #example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
    #example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
    #example["head_scores"] = head_scores.tolist()

    #parts = [words[m[0]:m[1]+1] for m in c]
    #parts = [[w for w in p if w != "\'s"] for p in parts]


    words = util.flatten(example["sentences"])
    mentions = [words[mention_starts[i]:1+mention_ends[i]] for i in range(len(mention_starts))]
    res_mentions = [" ".join([w for w in m if w != "\'s"]) for m in mentions]
    pr_antecedents = [words[mention_starts[predicted_antecedents[i]]:1+mention_ends[predicted_antecedents[i]]] if predicted_antecedents[i] != -1 else None for i in range(len(mention_starts))]
    res_pr_antecedents = [" ".join([w for w in m if w != "\'s"]) if m is not None else "" for m in pr_antecedents]

    #for i in range(len(mention_starts)) :
    #    print("{0} : {1} --- {2}".format(i, res_mentions[i], res_pr_antecedents[i]))

    example["mentions"] = res_mentions
    example["antecedents"] = res_pr_antecedents
    example["antecedent_scores"] = antecedent_scores
    return example

def get_predicted_antecedents(self, antecedents, antecedent_scores):

    predicted_antecedents = []

    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):

        if index < 0:

            predicted_antecedents.append(-1)

        else:

            predicted_antecedents.append(antecedents[i, index])

    return predicted_antecedents

def get_score_antecedents(text, model):
    example = create_example(text)
    tensorized_example = model.tensorize_example(example, is_training=False)
    feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
    _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

    predicted_scores = []

    for i, scores in enumerate(antecedent_scores):
        result = []
        for j, score in enumerate(scores) :
            if score > 0 :
                result.append((j-1, score))
        predicted_scores.append(result)

    predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

    words = util.flatten(example["sentences"])
    mentions = [words[mention_starts[i]:1+mention_ends[i]] for i in range(len(mention_starts))]
    res_mentions = [" ".join([w for w in m if w != "\'s"]) for m in mentions]
    pr_antecedents = [words[mention_starts[predicted_antecedents[i]]:1+mention_ends[predicted_antecedents[i]]] if predicted_antecedents[i] != -1 else None for i in range(len(mention_starts))]
    res_pr_antecedents = [" ".join([w for w in m if w != "\'s"]) if m is not None else "" for m in pr_antecedents]

    #for i in range(len(mention_starts)) :
    #    print("{0} : {1} --- {2}".format(i, res_mentions[i], res_pr_antecedents[i]))

    example["mentions"] = res_mentions
    example["antecedents"] = res_pr_antecedents
    example["antecedent_scores"] = antecedent_scores
    example["predicted_scores"] = predicted_scores

    return example



# def e2e_predict(df) :
#
#     with tf.Session() as session:
#         model.restore(session)
#
#         df['pred-e2e'] = df.apply(lambda row : e2e_predict_row(row), axis=1)
#
#         df["A-e2e"] = df['pred-e2e'][0]
#         df["B-e2e"] = df['pred-e2e'][1]
#         df["NEITHER-e2e"] = df['pred-e2e'][2]

config = util.initialize_from_env()
model = cm.CorefModel(config)

if __name__ == "__main__":


    train = pd.read_csv('../input/train.tsv', index_col=0, sep='\t').reset_index(drop=True)
    test = pd.read_csv('../input/test.tsv', index_col=0, sep='\t').reset_index(drop=True)

    with tf.Session() as session:
        model.restore(session)

        def predict_score(row) :
            text = row["Text"]

            l = row["Text"].rfind(".", 0, row["section_min"])
            r = row["Text"].find(".", row["section_max"])
            text = text[l+1:r]

            example = get_score_antecedents(text, model)
            mentions = example["mentions"]
            pr_antecedents = example["antecedents"]
            antecedent_scores = example["predicted_scores"]

            p_a = 0
            p_b = 0
            a_p = 0
            b_p = 0

            #print(mentions)

            for p_index in [index for index, value in enumerate(mentions) if value == row["Pronoun"]] :
                if len(antecedent_scores[p_index]) > 0 :
                    (p_ant_indexes, p_scores) = list(zip(*antecedent_scores[p_index]))
                else :
                    p_ant_indexes = []
                    p_scores = []

                for a_index in [index for index, value in enumerate(mentions) if value == row["A"]] :
                    if len(antecedent_scores[a_index]) > 0 :
                        (a_ant_indexes, a_scores) = list(zip(*antecedent_scores[a_index]))
                    else :
                        a_ant_indexes = []
                        a_scores = []

                    if a_index in p_ant_indexes and p_scores[p_ant_indexes.index(a_index)] > p_a :
                        p_a = p_scores[p_ant_indexes.index(a_index)]
                    if p_index in a_ant_indexes and a_scores[a_ant_indexes.index(p_index)] > a_p :
                        a_p = a_scores[a_ant_indexes.index(p_index)]

                for b_index in [index for index, value in enumerate(mentions) if value == row["B"]] :
                    if len(antecedent_scores[b_index]) > 0 :
                        (b_ant_indexes, b_scores) = list(zip(*antecedent_scores[b_index]))
                    else :
                        b_ant_indexes = []
                        b_scores = []

                    if b_index in p_ant_indexes and p_scores[p_ant_indexes.index(b_index)] > p_b :
                        p_b = p_scores[p_ant_indexes.index(b_index)]
                    if p_index in b_ant_indexes and b_scores[b_ant_indexes.index(p_index)] > b_p :
                        b_p = b_scores[b_ant_indexes.index(p_index)]

            #print(p_a, p_b, a_p, b_p)
            return (p_a, p_b, a_p, b_p)

        def predict_row(row) :
            text = row["Text"]

            l = row["Text"].rfind(".", 0, row["section_min"])
            r = row["Text"].find(".", row["section_max"])
            text = text[l+1:r]

            example = get_score_antecedents(text, model)
            mentions = example["mentions"]
            pr_antecedents = example["antecedents"]
            antecedent_scores = example["antecedent_scores"]

            a_coref = 0
            b_coref = 0

            if row["Pronoun"] in mentions :
                p_index = mentions.index(row["Pronoun"])

                if row["A"] in mentions :
                    a_index = mentions.index(row["A"])
                    if pr_antecedents[a_index] == row["Pronoun"] or pr_antecedents[p_index] == row["A"] :
                        a_coref = 1

                if row["B"] in mentions :
                    b_index = mentions.index(row["B"])
                    if pr_antecedents[b_index] == row["Pronoun"] or pr_antecedents[p_index] == row["B"] :
                        b_coref = 1

            return (a_coref, b_coref, min(1 - a_coref - b_coref, 0))



        def e2e_predict_row(row) :
            text = row["Text"]
            #print(text)
            #print("\n\t")
            l = row["Text"].rfind(".", 0, row["section_min"])
            r = row["Text"].find(".", row["section_max"])
            text = text[l+1:r]
            #print("\n\t")

            #print("----------------------------------")
            #print(text)
            #text = input("Document text: ")

            #text = input("Document text: ")
            example = make_predictions(text, model)
            a_coref = 0
            b_coref = 0
            #print_predictions(example)

            #print(row["A"])
            #print(row["B"])
            #print(row["Pronoun"])

            words = util.flatten(example["sentences"])

            for c in example["predicted_clusters"] :
                parts = [words[m[0]:m[1]+1] for m in c]
                parts = [[w for w in p if w != "\'s"] for p in parts]
                cluster = [" ".join(p) for p in parts]
                #print(cluster)

                if row["Pronoun"] not in cluster :
                    continue

                if row["A"] in cluster :
                    a_coref = 1
                else :
                    a_coref = 1
                    for a in row["A"].split() :
                        if a not in cluster :
                            a_coref = 0
                            break

                if row["B"] in cluster :
                    b_coref = 1
                else :
                    b_coref = 1
                    for b in row["B"].split() :
                        if b not in cluster :
                            b_coref = 0
                            break

                if (a_coref == 1) or (b_coref == 1) :
                    break
            #print("a = ", a_coref)
            #print("b = ", b_coref)

            return (a_coref, b_coref, min(1 - a_coref - b_coref, 0))

        def e2e_score_row(row) :
            text = row["Text"]
            #print(text)
            #print("\n\t")
            l = row["Text"].rfind(".", 0, row["section_min"])
            r = row["Text"].find(".", row["section_max"])
            text = text[l+1:r]

            example = make_predictions(text, model)

            words = util.flatten(example["sentences"])
            scores = util.flatten(example["head_scores"])

            #unreal = 10.0

            def get_score(column) :

                if row[column] in words :
                    return scores[words.index(row[column])]

                score = 0.0
                count = 0

                for w in row[column].split() :
                    if w in words :
                        score += scores[words.index(w)]
                        count +=1

                if count != 0 :
                    return score / count
                else :
                    return 0.0

            p_score = get_score("Pronoun")
            a_score = get_score("A")
            b_score = get_score("B")

            return (a_score, b_score, p_score)

        def predict_all(df, filename) :
            print("predict df")

            df['Pronoun-offset2'] = df['Pronoun-offset'] + df['Pronoun'].map(len)
            df['A-offset2'] = df['A-offset'] + df['A'].map(len)
            df['B-offset2'] = df['B-offset'] + df['B'].map(len)

            df['section_min'] = df[['Pronoun-offset', 'A-offset', 'B-offset']].min(axis=1)
            df['section_max'] = df[['Pronoun-offset2', 'A-offset2', 'B-offset2']].max(axis=1)

            df['A-dist'] = (df['Pronoun-offset'] - df['A-offset']).abs()
            df['B-dist'] = (df['Pronoun-offset'] - df['B-offset']).abs()

            df['pred-e2e-score'] = df.progress_apply(lambda row : predict_score(row), axis=1)
            print("score")

            df["p_a"] = df['pred-e2e-score'].apply(lambda x : x[0])
            df["p_b"] = df['pred-e2e-score'].apply(lambda x : x[1])
            df["a_p"] = df['pred-e2e-score'].apply(lambda x : x[2])
            df["b_p"] = df['pred-e2e-score'].apply(lambda x : x[3])

            df['pred-e2e'] = df.progress_apply(lambda row : e2e_predict_row(row), axis=1)
            df["A-e2e"] = df['pred-e2e'].apply(lambda x : x[0])
            df["B-e2e"] = df['pred-e2e'].apply(lambda x : x[1])
            df["NEITHER-e2e"] = df['pred-e2e'].apply(lambda x : x[2])
            print("lb 1")

            # df['pred-e2e-new'] = df.apply(lambda row : predict_row(row), axis=1)
            # df["A-e2e-new"] = df['pred-e2e-new'].apply(lambda x : x[0])
            # df["B-e2e-new"] = df['pred-e2e-new'].apply(lambda x : x[1])
            # df["NEITHER-e2e-new"] = df['pred-e2e-new'].apply(lambda x : x[2])
            # print("lb 2")

            renaming = {
                'p_a' : 'P-A-e2e', 'p_b' : 'P-B-e2e',
                'a_p' : 'A-P-e2e', 'b_p' : 'B-P-e2e'}

            df = df.rename(columns = renaming)

            df[['A-e2e', 'B-e2e', 'P-A-e2e', 'P-B-e2e', 'A-P-e2e', 'B-P-e2e']].to_csv(filename, sep='\t')

            return df

        e2e_test = predict_all(test, "../features/test_e2e.tsv")
        e2e_train = predict_all(train, "../features/train_e2e.tsv")