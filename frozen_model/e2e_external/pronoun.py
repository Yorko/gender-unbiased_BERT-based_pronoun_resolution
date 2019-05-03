from pronoun_cracker import *

def parse_json(embeddings):
    '''
    Parses the embeddigns given by BERT, and suitably formats them to be passed to the MLP model

    Input: embeddings, a DataFrame containing contextual embeddings from BERT, as well as the labels for the classification problem
    columns: "emb_A": contextual embedding for the word A
             "emb_B": contextual embedding for the word B
             "emb_P": contextual embedding for the pronoun
             "label": the answer to the coreference problem: "A", "B" or "NEITHER"

    Output: X, a numpy array containing, for each line in the GAP file, the concatenation of the embeddings of the target words
            Y, a numpy array containing, for each line in the GAP file, the one-hot encoded answer to the coreference problem
    '''
    embeddings.sort_index(inplace = True) # Sorting the DataFrame, because reading from the json file messed with the order
    X = np.zeros((len(embeddings),3*768))

    # Concatenate features
    for i in range(len(embeddings)):
        A = np.array(embeddings.loc[i,"emb_A"])
        B = np.array(embeddings.loc[i,"emb_B"])
        P = np.array(embeddings.loc[i,"emb_P"])
        X[i] = np.concatenate((A,B,P))

    return X

cracker = PronounCracker('pronoun', './input', './output')

# готовим датасет с нуля
#cracker.init_data()
#cracker.prepare_data()

cracker.load_data()
#cracker.prepare_new_data()

col = [ 'subj', 'obj', 'Pronoun-offset', 'A-offset', 'B-offset', 'section_min',
        'Pronoun-offset2', 'A-offset2', 'B-offset2', 'section_max', 'A-dist', 'B-dist', 'in_title']

col = ['Pronoun-offset', 'A-offset', 'B-offset', 'section_min',
       'Pronoun-offset2', 'A-offset2', 'B-offset2', 'section_max', 'A-poss', 'B-poss', 'A-dist', 'B-dist', 'in_title']

col = ['subj', 'obj', 'Pronoun-offset', 'A-offset', 'B-offset', 'section_min',
       'Pronoun-offset2', 'A-offset2', 'B-offset2', 'section_max', 'A-poss', 'B-poss', 'A-dist', 'B-dist',
       'in_title', 'Spacy-Coref-A', 'Spacy-Coref-B']

# все старое 0.7907 и 0.7847

#'Spacy-Coref-A', 'Spacy-Coref-B' докинуло до 0.7833 и 0.7783

col = ['subj', 'obj', 'Pronoun-offset', 'A-offset', 'B-offset', 'section_min',
       'Pronoun-offset2', 'A-offset2', 'B-offset2', 'section_max', 'A-poss', 'B-poss', 'A-dist', 'B-dist',
       'in_title', 'Spacy-Coref-A', 'Spacy-Coref-B', ]

# validation log_loss 0.6926779890556728
# test log_loss 0.6621221928255171


col = ['subj', 'obj', 'Pronoun-offset', 'A-offset', 'B-offset', 'section_min',
       'Pronoun-offset2', 'A-offset2', 'B-offset2', 'section_max', 'A-poss', 'B-poss', 'A-dist', 'B-dist',
       'in_title', 'Spacy-Coref-A', 'Spacy-Coref-B', 'A-e2e-new', 'B-e2e-new', 'NEITHER-e2e-new',
       'a_p', 'b_p', 'p_a', 'p_b', "A-stanford", "B-stanford"
       #'A-e2e-score', 'B-e2e-score', 'P-e2e-score',
       #"A-e2e-score-diff", "B-e2e-score-diff"
        ]

#cracker.fit(col)
#cracker.predict(col)

print(cracker.train.columns)

print("GB ONE")
cracker.fit_one(col)
cracker.predict_one(col)


print('test shape = ', cracker.test.shape)

development = pd.read_json(os.path.join(cracker.output_folder, "contextual_embeddings_gap_development.json"))
print('development shape = ', development.shape)
#print(development.columns)
#print(development.head())
X_development = parse_json(development)

val = pd.read_json(os.path.join(cracker.output_folder, "contextual_embeddings_gap_validation.json"))
print('val shape = ', development.shape)
X_val = parse_json(val)

test = pd.read_json(os.path.join(cracker.output_folder, "contextual_embeddings_gap_test.json"))
print('test shape = ', development.shape)
X_test = parse_json(test)

cracker.test_np = X_development
cracker.train_np = np.concatenate([X_test, X_val])
print('real train shape = ', cracker.train_np.shape)


print("BG EMB")
cracker.fit_emb(col)
cracker.predict_emb(col)

print("BG NN")
cracker.fit_predict_nn(col)


