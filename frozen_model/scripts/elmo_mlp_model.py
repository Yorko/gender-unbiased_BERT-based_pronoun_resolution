'''
Before running, download the files:
"https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
"https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
and place them in the folder:
../models/elmo

Requires 2.0.11 <= spacy <= 2.0.18

I ran into an issue using allennlp 0.8.3:
conflict caused by a folder and a file with
the same name "wikitable", located in
.local/lib/python3.6/site-packages/allennlp/data/dataset_readers/semantic_parsing

Tried different versions of allennlp, which created other problems.
To solve the conflict, I kept allennlp 0.8.3 and manually changed the name of the folder.
Let me know if you find a better solution.

To run ELMo on GPU, add parameter cuda_device = 0 to ElmoEmbedder
'''

import numpy as np, pandas as pd
import os
import zipfile
import sys
import time
import pickle

from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.tokenizers import word_tokenizer

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import log_loss

from keras import backend, models, layers, initializers, regularizers, constraints, optimizers
from keras import callbacks as kc
from keras import optimizers as ko

INPUT_PATH = "input/"
FEATURE_PATH = "features/"
MODEL_PATH = "models/elmo/"
TRAIN_FILE_PATH = INPUT_PATH + "train.tsv"
TEST_FILE_PATH = INPUT_PATH + "test.tsv"  # modify this path for stage 2

dense_layer_sizes = [37]
dropout_rate = 0.6
learning_rate = 0.001
n_fold = 5
batch_size = 32
epochs = 1000
patience = 100
lambd = 0.1  # L2 regularization


def get_elmo_fea(data, op, wg):
    '''
    Took this method from public kernel:
    https://www.kaggle.com/wochidadonggua/elmo-baseline

    modified it to concatenate all 3 layers
    '''

    def get_nearest(slot, target):
        for i in range(target, -1, -1):
            if i in slot:
                return i

    # add parameter cuda_device=0 to use GPU
    elmo = ElmoEmbedder(options_file=op, weight_file=wg)

    tk = word_tokenizer.WordTokenizer()
    tokens = tk.batch_tokenize(data.Text)
    idx = []

    for i in range(len(tokens)):
        idx.append([x.idx for x in tokens[i]])
        tokens[i] = [x.text for x in tokens[i]]

    vectors = elmo.embed_sentences(tokens)

    ans = []
    for i, vector in enumerate([v for v in vectors]):
        P_l = data.iloc[i].Pronoun
        A_l = data.iloc[i].A.split()
        B_l = data.iloc[i].B.split()

        P_offset = data.iloc[i]['Pronoun-offset']
        A_offset = data.iloc[i]['A-offset']
        B_offset = data.iloc[i]['B-offset']

        if P_offset not in idx[i]:
            P_offset = get_nearest(idx[i], P_offset)
        if A_offset not in idx[i]:
            A_offset = get_nearest(idx[i], A_offset)
        if B_offset not in idx[i]:
            B_offset = get_nearest(idx[i], B_offset)

        # P is a single token. For A and B, average over tokens in the span.
        emb_P = vector[:, idx[i].index(P_offset), :]
        emb_A = np.mean(vector[:, idx[i].index(A_offset):idx[i].index(A_offset) + len(A_l), :], axis=1)
        emb_B = np.mean(vector[:, idx[i].index(B_offset):idx[i].index(B_offset) + len(B_l), :], axis=1)

        ans.append(np.concatenate([emb_A[0], emb_A[1], emb_A[2], emb_B[0], emb_B[1], emb_B[2],
                                   emb_P[0], emb_P[1], emb_P[2]], axis=0).reshape(1, -1))

    emb = np.concatenate(ans, axis=0)
    return emb


def build_mlp_model(input_shape, seed):
    X_input = layers.Input(input_shape)

    # First dense layer
    X = layers.Dense(dense_layer_sizes[0], name='dense0', kernel_initializer=initializers.glorot_uniform(seed=seed))(
        X_input)
    X = layers.BatchNormalization(name='bn0')(X)
    X = layers.Activation('relu')(X)
    X = layers.Dropout(dropout_rate, seed=seed)(X)

    # Second dense layer
    # X = layers.Dense(dense_layer_sizes[0], name = 'dense1', kernel_initializer=initializers.glorot_uniform(seed=seed))(X)
    # X = layers.BatchNormalization(name = 'bn1')(X)
    # X = layers.Activation('relu')(X)
    # X = layers.Dropout(dropout_rate, seed = seed)(X)

    # Output layer
    X = layers.Dense(3, name='output', kernel_regularizer=regularizers.l2(lambd),
                     kernel_initializer=initializers.glorot_uniform(seed=seed))(X)
    X = layers.Activation('softmax')(X)

    # Create model
    model = models.Model(input=X_input, output=X, name="classif_model")
    return model


def build_elmo_embeddings():
    op = MODEL_PATH + "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    wg = MODEL_PATH + "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    print("Started at ", time.ctime())
    test_data = pd.read_csv(TEST_FILE_PATH, sep='\t')
    X_test = get_elmo_fea(test_data, op, wg)

    train_data = pd.read_csv(TRAIN_FILE_PATH, sep='\t')
    X_train = get_elmo_fea(train_data, op, wg)
    print("Finished at ", time.ctime())

    elmo_train = pd.DataFrame(X_train)
    elmo_test = pd.DataFrame(X_test)

    elmo_train.to_csv(FEATURE_PATH + "train_elmo_embedding.csv", index=False)
    elmo_test.to_csv(FEATURE_PATH + "test_elmo_embedding.csv", index=False)


def get_labels_for_elmo():
    train = pd.read_csv(TRAIN_FILE_PATH, sep='\t')

    Y_train = np.zeros((len(train), 3))

    for i in range(len(train)):
        if train.loc[i, "A-coref"]:
            Y_train[i, 0] = 1
        elif train.loc[i, "B-coref"]:
            Y_train[i, 1] = 1
        else:
            Y_train[i, 2] = 1

    return Y_train


def train_elmo_mlp_model(test=False):
    '''
    Runs 5-fold CV and blending over 3 seed values
    Simple MLP architecture: 1 hidden layer seems to work better than 2
    '''

    # Read the embeddings from file
    X_train = pd.read_csv(FEATURE_PATH + "train_elmo_embedding.csv").values
    X_test = pd.read_csv(FEATURE_PATH + "test_elmo_embedding.csv").values

    # Get the labels to train the MLP model
    Y_train = get_labels_for_elmo()

    # Initializing the predictions
    prediction = np.zeros((len(X_test), 3))
    oof = np.zeros((len(X_train), 3))

    # Training and cross-validation
    scores = []
    seed_list = [1, 6003, 10000007]

    for seed in seed_list:
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

        for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
            # split training and validation data
            print('Fold', fold_n, 'started at', time.ctime())
            X_tr, X_val = X_train[train_index], X_train[valid_index]
            Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]

            # Define the model, re-initializing for each fold
            classif_model = build_mlp_model([X_train.shape[1]], seed)
            classif_model.compile(optimizer=optimizers.Adam(lr=learning_rate),
                                  loss="categorical_crossentropy")
            callbacks = [kc.EarlyStopping(monitor='val_loss', patience=patience,
                                          restore_best_weights=True)]

            # train the model
            classif_model.fit(x=X_tr, y=Y_tr, epochs=epochs, batch_size=batch_size,
                              callbacks=callbacks, validation_data=(X_val, Y_val), verbose=0)

            # make predictions on validation and test data
            pred_valid = classif_model.predict(x=X_val, verbose=0)
            pred = classif_model.predict(x=X_test, verbose=0)

            scores.append(log_loss(Y_val, pred_valid))
            prediction += pred
            oof[valid_index] += pred_valid
    prediction /= n_fold * len(seed_list)
    oof /= len(seed_list)

    # Print CV scores, as well as score on the test data
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    print(scores)
    # print("Test score:", log_loss(Y_test,prediction))

    df_pred = pd.DataFrame(prediction, columns=["elmo_A", "elmo_B", "elmo_N"]).round(4)
    df_oof = pd.DataFrame(oof, columns=["elmo_A", "elmo_B", "elmo_N"]).round(4)

    df_oof.to_csv(FEATURE_PATH + "train_elmo_prob.tsv", sep='\t')
    df_pred.to_csv(FEATURE_PATH + "test_elmo_prob.tsv", sep='\t')


def build_train_elmo():
    build_elmo_embeddings()
    train_elmo_mlp_model()

