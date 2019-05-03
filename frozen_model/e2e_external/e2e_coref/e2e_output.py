
import sys
sys.path.append('..')
from pronoun_cracker import *

import numpy as np
import pandas as pd

cracker = PronounCracker('pronoun', '../input', '../output')
cracker.load_data()

print(cracker.train.columns)

renaming = {
    'p_a' : 'P-A-e2e', 'p_b' : 'P-B-e2e',
    'a_p' : 'A-P-e2e', 'b_p' : 'B-P-e2e'}

train = cracker.train.rename(columns = renaming)
test = cracker.test.rename(columns = renaming)

print(train[['A-e2e', 'B-e2e', 'P-A-e2e', 'P-B-e2e', 'A-P-e2e', 'B-P-e2e']].head())
train[['A-e2e', 'B-e2e', 'P-A-e2e', 'P-B-e2e', 'A-P-e2e', 'B-P-e2e']].to_csv("train_e2e.tsv", sep='\t')
test[['A-e2e', 'B-e2e', 'P-A-e2e', 'P-B-e2e', 'A-P-e2e', 'B-P-e2e']].to_csv("test_e2e.tsv", sep='\t')

