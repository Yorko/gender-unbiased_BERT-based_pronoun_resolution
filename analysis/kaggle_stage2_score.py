import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from pathlib import Path

SUB_PATH = Path("submissions")
INPUT_PATH = Path("input")


def read_predictions():
    # BERT fine-tuning predictions
    ft1 = pd.read_csv(SUB_PATH/"gap_out2_12801_kenkrige_results.csv", index_col = "ID")
    ft2 = pd.read_csv(SUB_PATH/"gap_out2_12803_kenkrige_results.csv", index_col = "ID")
    ft3 = pd.read_csv(SUB_PATH/"gap_out2_12809_kenkrige_results.csv", index_col = "ID")
    ft4 = pd.read_csv( SUB_PATH/"gap_out2_6401_kenkrige_results.csv", index_col = "ID")
    ft5 = pd.read_csv( SUB_PATH/"gap_out2_6409_kenkrige_results.csv", index_col = "ID")
    ft6 = pd.read_csv( SUB_PATH/"gap_out2_6419_kenkrige_results.csv", index_col = "ID")

    ft = (ft1 + ft2 + ft3 + ft4 + ft5 + ft6)/6

    # BERT frozen predictions
    fz = pd.read_csv(SUB_PATH/"submission_fivezeros_final_2.csv", index_col = "ID")


    # blend model
    bl = 0.65 * ft + 0.35 * fz

    return ft, fz, bl 


def score_predictions(ft, fz, bl):
    # read true labels for kaggle stage2 test file
    test = pd.read_csv(INPUT_PATH/"solution_stage_2.csv", index_col = "ID")

    # select the 760 entries used for scoring
    # the rest have dummy labels and are not meaningful for scoring
    private_ind = [ind for ind in test.index if test.loc[ind, "Usage"] != "Ignored"]
    test = test.loc[private_ind, ["A", "B", "NEITHER"]]
    ft = ft.loc[private_ind]
    fz = fz.loc[private_ind]
    bl = bl.loc[private_ind]

    # compute log loss
    print("Fine-tuned model has logarithmic loss ", round(log_loss(test, ft),3))
    print("Frozen model has logarithmic loss ", round(log_loss(test, fz),3))
    print("Blended model has logarithmic loss ", round(log_loss(test, bl),3))


def main():
    ft, fz, bl = read_predictions()
    score_predictions(ft, fz, bl)


if __name__ == '__main__':
    main()