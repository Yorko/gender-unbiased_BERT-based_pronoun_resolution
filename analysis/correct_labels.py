import numpy as np, pandas as pd 

INPUT_PATH = "../input/"

# read test set
test = pd.read_csv(INPUT_PATH + "gap-test.tsv", sep = '\t', index_col = "ID")

# read corrections
corrections = pd.read_csv(INPUT_PATH + "corrections.csv")
corrections.columns = ["ID", "label"]
corrections = corrections.set_index("ID")

# select corrections for test set
ID_to_correct = [ind for ind in corrections.index if "test" in ind]
ID_to_A = [ind for ind in ID_to_correct if corrections.loc[ind, "label"] == "A"]
ID_to_B = [ind for ind in ID_to_correct if corrections.loc[ind, "label"] == "B"]

# correct labels
test.loc[ID_to_correct, ["A-coref", "B-coref"]] = False
test.loc[ID_to_A, "A-coref"] = True
test.loc[ID_to_B, "B-coref"] = True

# output corrected test set
test.to_csv(INPUT_PATH + "gap-test-corrected.tsv", sep = '\t')