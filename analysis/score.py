import numpy as np, pandas as pd 
pd.set_option("display.precision", 4)
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import argparse
from pathlib import Path

SUB_PATH = Path("submissions")
INPUT_PATH = Path("input")
IMG_PATH = Path("")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', '--test_file_name', type=str, default='gap-test.tsv',
                        help='Filename to read true labels from')
    parser.add_argument('-result', '--result_file_name', type=str, default='result.csv',
                        help='Filename to write results to')
    return parser.parse_args()


def read_predictions():
    # BERT fine-tuning predictions
    ft1 = pd.read_csv(SUB_PATH/"gap_paper_out_12802_kenkrige_results.csv").drop(columns = "ID")
    ft2 = pd.read_csv(SUB_PATH/"gap_paper_out_12803_kenkrige_results.csv").drop(columns = "ID")
    ft3 = pd.read_csv(SUB_PATH/"gap_paper_out_12804_kenkrige_results.csv").drop(columns = "ID")
    ft4 = pd.read_csv( SUB_PATH/"gap_paper_out_6400_kenkrige_results.csv").drop(columns = "ID")
    ft5 = pd.read_csv( SUB_PATH/"gap_paper_out_6402_kenkrige_results.csv").drop(columns = "ID")
    ft6 = pd.read_csv( SUB_PATH/"gap_paper_out_6404_kenkrige_results.csv").drop(columns = "ID")

    ft = (ft1 + ft2 + ft3 + ft4 + ft5 + ft6)/6

    # clip fine-tuned predictions
    threshold = 10**(-2)
    ft[ft < threshold] = threshold
    #ft = ft / np.sum(ft.values, axis=1, keepdims=True)

    # BERT frozen predictions
    fz = pd.read_csv(SUB_PATH/"pred1_for_gap_test.csv").drop(columns = "ID")

    return ft, fz


def read_ground_truth(test_file):
    # read the test file with ground truth labels
    test = pd.read_csv(test_file, sep = '\t')

    A_test = test["A-coref"].astype(int)
    B_test = test["B-coref"].astype(int)
    N_test = 1 - A_test - B_test
    Y_test = pd.concat([A_test, B_test, N_test], axis = 1)

    Y_test.columns = ["A", "B", "N"]

    female_pron = ["she", "her", "hers"]
    male_pron = ["he", "him", "his"]

    # Boolean variable for gender
    # In gap-test.tsv there are 1000 M/1000 F
    Y_test["female"] = test.apply(lambda row: row["Pronoun"].lower() in female_pron, axis = 1)

    return test, Y_test


def blend_concat(ft, fz, Y_test):
    # make a blend of the fine-tuned and frozen models
    blend = 0.5 * ft + 0.5 * fz

    ft.columns = ["ft-A", "ft-B", "ft-N"]
    fz.columns = ["fz-A", "fz-B", "fz-N"]
    blend.columns = ["bl-A", "bl-B", "bl-N"]

    # save a DataFrame with ground truth, and
    # probabilities predicted by all 3 models
    pred = pd.concat([Y_test,ft,fz,blend], axis = 1)

    return pred


def row_logloss(row, model):
    # Compute logarithmic loss for a single 3-class prediction
    y = np.array([row["A"], row["B"], row["N"]]).reshape(1,-1)
    pred = np.array([row[model + "-A"], row[model + "-B"], row[model + "-N"]]).reshape(1,-1)

    return log_loss(y,pred)


def row_accuracy(row, model):
    # Compute accuracy for a single 3-class prediction
    y = np.array([row["A"], row["B"], row["N"]])
    pred = np.array([row[model + "-A"], row[model + "-B"], row[model + "-N"]])

    return y[np.argmax(pred)]


def row_confusion(row, model):
    # Computes entries of confusion matrix (TP, FP, FN, TN)
    # for a single 2-class prediction

    arg = row[model + "_arg"]

    TP, FP , FN, TN = 0, 0, 0, 0

    for name in ["A", "B"]:
        if arg == name: # prediction positive
            if row[name]: TP += 1 # true positive
            else: FP +=1 # false positive
        else: # prediction negative
            if row[name]: FN +=1 # false negative
            else: TN +=1 # true negative

    return pd.Series([TP, FP, FN, TN])


def compute_metrics(pred):
    # logloss and accuracy are computed for 3 classes: A/B/Neither
    # F1 is computed for 2 classes: A/not A, B/not B, following methodology of GAP authors

    models = ["ft", "fz", "bl"]
    metrics = ["logloss-M", "logloss-F", "logloss-O", "logloss-B", 
               "accuracy-M", "accuracy-F", "accuracy-O", "accuracy-B",
               "F1-M", "F1-F", "F1-O", "F1-B"]

    for model in models:
        pred[model + "_logloss"] = pred.apply(lambda row: row_logloss(row,model), axis = 1)
        pred[model + "_accuracy"] = pred.apply(lambda row: row_accuracy(row,model), axis = 1)

        # given 3-class probabilities, take argmax to obtain prediction
        pred[model + "_arg"] = np.argmax(pred[[model + "-A", model + "-B", model + "-N"]].values, axis = 1)
        pred[model + "_arg"] = pred[model + "_arg"].replace({0:"A", 1:"B", 2:"N"})

        pred[[model + "_TP", model + "_FP", model + "_FN", model + "_TN"]] = pred.apply(lambda row: row_confusion(row,model), axis = 1)

    results = pd.DataFrame(index = models, columns = metrics, dtype = 'float')

    for model in models:
        results.loc[model, "logloss-O"] = np.mean(pred[model + "_logloss"])
        results.loc[model, "logloss-F"] = np.mean(pred.loc[ pred["female"] == True, model + "_logloss" ])
        results.loc[model, "logloss-M"] = np.mean(pred.loc[ pred["female"] == False, model + "_logloss" ])

        results.loc[model, "accuracy-O"] = np.mean(pred[model + "_accuracy"])
        results.loc[model, "accuracy-F"] = np.mean(pred.loc[ pred["female"] == True, model + "_accuracy" ])
        results.loc[model, "accuracy-M"] = np.mean(pred.loc[ pred["female"] == False, model + "_accuracy" ])

        TP_F = np.sum(pred.loc[ pred["female"] == True, model + "_TP" ].values)
        TP_M = np.sum(pred.loc[ pred["female"] == False, model + "_TP" ].values)
        TP = TP_M + TP_F

        FP_F = np.sum(pred.loc[ pred["female"] == True, model + "_FP" ].values)
        FP_M = np.sum(pred.loc[ pred["female"] == False, model + "_FP" ].values)
        FP = FP_M + FP_F

        FN_F = np.sum(pred.loc[ pred["female"] == True, model + "_FN" ].values)
        FN_M = np.sum(pred.loc[ pred["female"] == False, model + "_FN" ].values)
        FN = FN_M + FN_F

        precision_F = TP_F / (TP_F + FP_F)
        precision_M = TP_M / (TP_M + FP_M)
        precision = TP / (TP + FP)

        recall_F = TP_F / (TP_F + FN_F)
        recall_M = TP_M / (TP_M + FN_M)
        recall = TP / (TP + FN)

        results.loc[model, "F1-O"] = 2 * precision * recall / (precision + recall)
        results.loc[model, "F1-F"] = 2 * precision_F * recall_F / (precision_F + recall_F)
        results.loc[model, "F1-M"] = 2 * precision_M * recall_M / (precision_M + recall_M)                



    results["logloss-B"] = results["logloss-M"] / results["logloss-F"]
    results["accuracy-B"] = results["accuracy-F"] / results["accuracy-M"]
    results["F1-B"] = results["F1-F"] / results["F1-M"]

    results.index = ["fine-tuned", "frozen", "blend"]
    results = results.round(3)

    return pred, results


def save_tables(results):
    metrics = ["logloss", "accuracy", "F1"]
    for metric in metrics:
        cols = [col for col in results.columns if metric in col]
        df = results[cols]
        df.columns = ["Male", "Female", "Overall", "Bias"]

        plt.figure(figsize = (7,7))
        ax = plt.subplot( frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.table(ax, df)  # where df is your data frame
        plt.savefig(IMG_PATH + metric + '.png')


def main(test_file, out_file):
    ft, fz = read_predictions()
    test_df, Y_test = read_ground_truth(test_file)
    pred = blend_concat(ft,fz,Y_test)
    pred, results = compute_metrics(pred)
    results.to_csv(out_file)
    print(results)
    #save_tables(results)

if __name__ == '__main__':

    args = parse_args()

    main(test_file=INPUT_PATH/args.test_file_name, out_file=args.result_file_name)
