import numpy as np
import pandas as pd

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
import nltk
from sklearn import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
#import xgboost as xgb
#from xgboost import XGBClassifier


import tensorflow as tf

from keras import backend, models, layers, initializers, regularizers, constraints, optimizers
from keras import callbacks as kc
from keras import optimizers as ko

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import log_loss
import time

dense_layer_sizes = [37]
dropout_rate = 0.6
learning_rate = 0.001
n_fold = 5
batch_size = 32
epochs = 1000
patience = 100
# n_test = 100
lambd = 0.1 # L2 regularization

def build_mlp_model(input_shape):
    X_input = layers.Input(input_shape)

    # First dense layer
    X = layers.Dense(dense_layer_sizes[0], name = 'dense0')(X_input)
    X = layers.BatchNormalization(name = 'bn0')(X)
    X = layers.Activation('relu')(X)
    X = layers.Dropout(dropout_rate, seed = 7)(X)

    # Second dense layer
    # 	X = layers.Dense(dense_layer_sizes[0], name = 'dense1')(X)
    # 	X = layers.BatchNormalization(name = 'bn1')(X)
    # 	X = layers.Activation('relu')(X)
    # 	X = layers.Dropout(dropout_rate, seed = 9)(X)

    # Output layer
    X = layers.Dense(3, name = 'output', kernel_regularizer = regularizers.l2(lambd))(X)
    X = layers.Activation('softmax')(X)

    # Create model
    model = models.Model(input = X_input, output = X, name = "classif_model")
    return model


from cracker import *

def name_replace(s, r1, r2):
    s = str(s).replace(r1,r2)
    for r3 in r1.split(' '):
        s = str(s).replace(r3,r2)
    return s

def scrape_url(url):
    '''
    get the title of the wikipedia page and replace "_" with white space
    '''
    return url[29:].lower().replace("_"," ")

def check_name_in_string(name,string):
    '''
    check whether the name string is a substring of another string (i.e. wikipedia title)
    '''

    return name.lower() in string

def predict_coref(df):
    pred =[]
    for index, row in df.iterrows():
        wiki_title = scrape_url(row["URL"])
        if (check_name_in_string(row["A"],wiki_title)):
            pred.append(0)
        else:
            if (check_name_in_string(row["B"],wiki_title)):
                pred.append(1)
            else:
                pred.append(2)
    return pred

def get_nlp_features(s, w):
    doc = nlp(str(s))
    tokens = pd.DataFrame([[token.text, token.dep_] for token in doc], columns=['text', 'dep'])
    return len(tokens[((tokens['text']==w) & (tokens['dep']=='poss'))])

def get_coref(row):
    coref = None

    nlpr = nlp(row['Text_unnamed'])

    # dunno if more direct way to get token from text offset
    for tok in nlpr.doc:
        if tok.idx == row['Pronoun-offset']:
            # model limitation that sometimes there are no coref clusters for the token?
            # also, sometimes the coref clusters will just be something like:
            # He: his, him, his
            # So there is no proper name to map back to?
            try:
                if len(tok._.coref_clusters) > 0:
                    coref = tok._.coref_clusters[0][0].text
            except:
                # for some, get the following exception just checking len(tok._.coref_clusters)
                # *** TypeError: 'NoneType' object is not iterable
                pass
            break

    if coref:
        coref = coref.lower()
        # sometimes the coref is I think meant to be the same as A or B, but
        # it is either a substring or superstring of A or B
        A_Noun = row['A'].lower()
        B_Noun = row['B'].lower()
        if coref in A_Noun or A_Noun in coref:
            coref = A_Noun
        elif coref in B_Noun or B_Noun in coref:
            coref = B_Noun

    return coref


class PronounCracker(Cracker) :

    def __init__(self, name, data_folder = "./input", output_folder = "./output") :

        Cracker.__init__(self, name, data_folder, output_folder)
        print('pronoun cracker constructor ', name)

    def init_data(self):
        print('load data')

        gh_test = pd.read_csv(os.path.join(self.data_folder, "gap-test.tsv"), delimiter='\t')
        #print('gh_test columns = ', gh_test.columns)
        #print(gh_test.head())
        gh_valid = pd.read_csv(os.path.join(self.data_folder, "gap-validation.tsv"), delimiter='\t')
        #print(gh_valid.head())
        self.train = pd.concat((gh_test, gh_valid)).reset_index(drop=True)
        #.rename(columns={'A': 'A_Noun', 'B': 'B_Noun'}).reset_index(drop=True)

        #self.test = pd.read_csv(os.path.join(self.data_folder, "test_stage_1.tsv"), delimiter='\t')
        self.test = pd.read_csv(os.path.join(self.data_folder, "gap-development.tsv"), delimiter='\t')
        #print(self.test.shape)
        #print(self.test.head())
        #print('test stage 1 columns = ', self.test.columns)
        #print(self.test.head())
        #sub = pd.read_csv('../input/sample_submission_stage_1.csv')

    def make_features(self, df):
        df['A-coref'] = df['A-coref'].astype(int)
        df['B-coref'] = df['B-coref'].astype(int)
        df['NEITHER-coref'] = 1.0 - (df['A-coref'] + df['B-coref'])

        df['Pronoun-offset2'] = df['Pronoun-offset'] + df['Pronoun'].map(len)
        df['A-offset2'] = df['A-offset'] + df['A'].map(len)
        df['B-offset2'] = df['B-offset'] + df['B'].map(len)

        df['section_min'] = df[['Pronoun-offset', 'A-offset', 'B-offset']].min(axis=1)
        df['section_max'] = df[['Pronoun-offset2', 'A-offset2', 'B-offset2']].max(axis=1)

        df['A-dist'] = (df['Pronoun-offset'] - df['A-offset']).abs()
        df['B-dist'] = (df['Pronoun-offset'] - df['B-offset']).abs()

        df['male'] = df['Pronoun'].str.lower().apply(lambda x : 1 if x in ['he', 'his', 'him'] else 0)
        df['subj'] = df['Pronoun'].str.lower().apply(lambda x : 1 if x in ['he', 'she'] else 0)
        df['obj'] = df['Pronoun'].str.lower().apply(lambda x : 1 if x in ['him', 'her'] else 0)
        df['poss'] = df['Pronoun'].str.lower().apply(lambda x : 1 if x in ['his', 'hers'] else 0)

        df['in_title'] = predict_coref(df)

        #df['Text'] = df.apply(lambda r: r['Text'][: r['Pronoun-offset']] + 'pronountarget' + r['Text'][r['Pronoun-offset'] + len(str(r['Pronoun'])): ], axis=1)
        df['Text_unnamed'] = df.apply(lambda r: name_replace(r['Text'], r['A'], 'subjectone'), axis=1)
        df['Text_unnamed'] = df.apply(lambda r: name_replace(r['Text_unnamed'], r['B'], 'subjecttwo'), axis=1)

        df['A-poss'] = df['Text_unnamed'].map(lambda x: get_nlp_features(x, 'subjectone'))
        df['B-poss'] = df['Text_unnamed'].map(lambda x: get_nlp_features(x, 'subjecttwo'))

        df['Coref'] = df.apply(get_coref, axis=1)
        df['Spacy-Coref-A'] = df['Coref'] == df['A'].str.lower()
        df['Spacy-Coref-B'] = df['Coref'] == df['B'].str.lower()

        return(df)

    def make_new_features(self, df):
        print('make new features')

    def fit(self, col):
        x1, x2, y1, y2 = model_selection.train_test_split(self.train[col].fillna(-1),
                self.train[['A-coref', 'B-coref', 'NEITHER-coref']], test_size=0.2, random_state=1)

        m = ensemble.RandomForestClassifier(max_depth = 7, n_estimators=1000, random_state=33)
        self.model = multiclass.OneVsRestClassifier(m)
        #self.model = ensemble.RandomForestClassifier(max_depth = 7, n_estimators=1000, random_state=33)

        self.model.fit(x1, y1)
        print('validation log_loss', metrics.log_loss(y2, self.model.predict_proba(x2)))
        self.model.fit(self.train[col].fillna(-1), self.train[['A-coref', 'B-coref', 'NEITHER-coref']])
        #print('importances = ', m.feature_importances_)

    def predict(self, col):
        results = self.model.predict_proba(self.test[col])
        self.test['A-pred'] = results[:,0]
        self.test['B-pred'] = results[:,1]
        self.test['NEITHER-pred'] = results[:,2]

        #self.test[['ID', 'A-pred', 'B-pred', 'NEITHER-pred']].to_csv(os.path.join(self.output_folder, 'submission.csv'), index=False)

        print('test log_loss', metrics.log_loss(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results))
        print("----------------------------------")
        results = self.model.predict(self.test[col])
        print(self.test[['A-coref', 'B-coref', 'NEITHER-coref']].head())
        print(results)
        print('accuracy', metrics.accuracy_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results))
        print('test precision macro', metrics.precision_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "macro"))
        print('test precision micro', metrics.precision_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "micro"))
        print('test F1 macro', metrics.f1_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "macro"))
        print('test F1 micro', metrics.f1_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "micro"))

    def fit_one(self, col):
        self.train['coref-true'] = self.train.apply(lambda row : 1 if row['A-coref'] == 1 else (2 if row['B-coref'] == 1 else 0), axis=1)
        self.test['coref-true'] = self.train.apply(lambda row : 1 if row['A-coref'] == 1 else (2 if row['B-coref'] == 1 else 0), axis=1)

        x1, x2, y1, y2 = model_selection.train_test_split(self.train[col].fillna(-1),
                                                          self.train['coref-true'], test_size=0.2, random_state=1)

        #m = ensemble.RandomForestClassifier(max_depth = 7, n_estimators=1000, random_state=33)
        #self.model = multiclass.OneVsRestClassifier(m)
        #self.model = ensemble.RandomForestClassifier(max_depth = 7, n_estimators=1000, random_state=33)
        self.model = GradientBoostingClassifier(random_state=33)
        #self.model = RandomForestClassifier(random_state=33)

        self.model.fit(x1, y1)
        print('validation log_loss', metrics.log_loss(y2, self.model.predict_proba(x2)))
        self.model.fit(self.train[col].fillna(-1), self.train['coref-true'])
        print('importances = ', self.model.feature_importances_)

    def predict_one(self, col):
        results = self.model.predict_proba(self.test[col])

        print(results)

        self.test['A-pred'] = results[:,1]
        self.test['B-pred'] = results[:,2]
        self.test['NEITHER-pred'] = results[:,0]

        #self.test[['ID', 'A-pred', 'B-pred', 'NEITHER-pred']].to_csv(os.path.join(self.output_folder, 'submission.csv'), index=False)

        print('test log_loss', metrics.log_loss(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], self.test[['A-pred', 'B-pred', 'NEITHER-pred']]))
        print("----------------------------------")
        results = self.model.predict(self.test[col])

        #
        # print(self.test[['A-coref', 'B-coref', 'NEITHER-coref']].head())
        # print(results)
        # print('accuracy', metrics.accuracy_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results))
        # print('test precision macro', metrics.precision_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "macro"))
        # print('test precision micro', metrics.precision_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "micro"))
        # print('test F1 macro', metrics.f1_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "macro"))
        # print('test F1 micro', metrics.f1_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "micro"))

    def fit_emb(self, col):
        self.train['coref-true'] = self.train.apply(lambda row : 1 if row['A-coref'] == 1 else (2 if row['B-coref'] == 1 else 0), axis=1)
        self.test['coref-true'] = self.train.apply(lambda row : 1 if row['A-coref'] == 1 else (2 if row['B-coref'] == 1 else 0), axis=1)

        remove_train = [row for row in range(len(self.train_np)) if np.sum(np.isnan(self.train_np[row]))]
        self.train_np[remove_train] = np.zeros(3*768)
        print('remove train shape = ', len(remove_train))
        #X_train = np.delete(self.train_np, remove_train, 0)
        #Y_train = np.delete(self.train['coref-true'], remove_train, 0)

        X_train = np.concatenate((self.train_np, self.train[col].fillna(-1).values), axis=1)
        Y_train = self.train['coref-true'].values

        x1, x2, y1, y2 = model_selection.train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

        #m = ensemble.RandomForestClassifier(max_depth = 7, n_estimators=1000, random_state=33)
        #self.model = multiclass.OneVsRestClassifier(m)
        #self.model = ensemble.RandomForestClassifier(max_depth = 7, n_estimators=1000, random_state=33)
        self.model = GradientBoostingClassifier(random_state=33)
        #self.model = RandomForestClassifier(random_state=33)

        self.model.fit(x1, y1)
        print('validation log_loss', metrics.log_loss(y2, self.model.predict_proba(x2)))
        self.model.fit(X_train, Y_train)
        print('importances = ', self.model.feature_importances_)

    def predict_emb(self, col):

        # We want predictions for all development rows. So instead of removing rows, make them 0
        remove_test = [row for row in range(len(self.test_np)) if np.sum(np.isnan(self.test_np[row]))]
        self.test_np[remove_test] = np.zeros(3*768)
        X_test = np.concatenate((self.test_np, self.test[col].fillna(-1).values), axis=1)
        results = self.model.predict_proba(X_test)

        #print(results)

        self.test['A-pred'] = results[:,1]
        self.test['B-pred'] = results[:,2]
        self.test['NEITHER-pred'] = results[:,0]

        self.test[['ID', 'A-pred', 'B-pred', 'NEITHER-pred']].to_csv(os.path.join(self.output_folder, 'submission.csv'), index=False)

        print('test log_loss', metrics.log_loss(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], self.test[['A-pred', 'B-pred', 'NEITHER-pred']]))
        print("----------------------------------")
        results = self.model.predict(X_test)

        #
        # print(self.test[['A-coref', 'B-coref', 'NEITHER-coref']].head())
        # print(results)
        # print('accuracy', metrics.accuracy_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results))
        # print('test precision macro', metrics.precision_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "macro"))
        # print('test precision micro', metrics.precision_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "micro"))
        # print('test F1 macro', metrics.f1_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "macro"))
        # print('test F1 micro', metrics.f1_score(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], results, average = "micro"))

    def fit_predict_nn(self, col):
        print(self.train['NEITHER-coref'].value_counts())
        self.train['coref-true'] = self.train.apply(lambda row : 1 if row['A-coref'] == 1 else (2 if row['B-coref'] == 1 else 0), axis=1)
        self.test['coref-true'] = self.train.apply(lambda row : 1 if row['A-coref'] == 1 else (2 if row['B-coref'] == 1 else 0), axis=1)

        remove_train = [row for row in range(len(self.train_np)) if np.sum(np.isnan(self.train_np[row]))]
        self.train_np[remove_train] = np.zeros(3*768)
        print('remove train shape = ', len(remove_train))
        #X_train = np.delete(self.train_np, remove_train, 0)
        #Y_train = np.delete(self.train['coref-true'], remove_train, 0)

        X_train = np.concatenate((self.train_np, self.train[col].fillna(-1).values), axis=1)
        Y_train = self.train[['A-coref', 'B-coref', 'NEITHER-coref']].values

        remove_test = [row for row in range(len(self.test_np)) if np.sum(np.isnan(self.test_np[row]))]
        self.test_np[remove_test] = np.zeros(3*768)
        X_test = np.concatenate((self.test_np, self.test[col].fillna(-1).values), axis=1)
        Y_test = self.test[['A-coref', 'B-coref', 'NEITHER-coref']].values

        prediction = np.zeros((len(X_test),3)) # testing predictions

        # Training and cross-validation
        folds = KFold(n_splits=n_fold, shuffle=True, random_state=3)
        scores = []
        for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
            # split training and validation data
            print('Fold', fold_n, 'started at', time.ctime())
            X_tr, X_val = X_train[train_index], X_train[valid_index]
            Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]

            # Define the model, re-initializing for each fold
            classif_model = build_mlp_model([X_train.shape[1]])
            classif_model.compile(optimizer = optimizers.Adam(lr = learning_rate), loss = "categorical_crossentropy")
            callbacks = [kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights = True)]

            # train the model
            classif_model.fit(x = X_tr, y = Y_tr, epochs = epochs, batch_size = batch_size, callbacks = callbacks, validation_data = (X_val, Y_val), verbose = 0)

            # make predictions on validation and test data
            pred_valid = classif_model.predict(x = X_val, verbose = 0)
            pred = classif_model.predict(x = X_test, verbose = 0)

            # oof[valid_index] = pred_valid.reshape(-1,)
            scores.append(log_loss(Y_val, pred_valid))
            prediction += pred
        prediction /= n_fold


        # Print CV scores, as well as score on the test data
        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
        print(scores)
        print("Test score:", log_loss(Y_test,prediction))

        self.test['A-pred'] = prediction[:,0]
        self.test['B-pred'] = prediction[:,1]
        self.test['NEITHER-pred'] = prediction[:,2]

        self.test[['ID', 'A-pred', 'B-pred', 'NEITHER-pred']].to_csv(os.path.join(self.output_folder, 'submission.csv'), index=False)

        print('test log_loss', metrics.log_loss(self.test[['A-coref', 'B-coref', 'NEITHER-coref']], self.test[['A-pred', 'B-pred', 'NEITHER-pred']]))
