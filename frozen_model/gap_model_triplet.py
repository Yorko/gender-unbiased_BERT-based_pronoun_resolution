import pandas as pd
from tqdm import tqdm
import json
import time
import os
from keras import backend, models, layers, initializers, regularizers, constraints, optimizers
from keras import callbacks as kc
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GroupKFold
from sklearn.metrics import log_loss
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
from tensorflow import set_random_seed
set_random_seed(17)
np.random.seed(17)


class GAPModelTriplet(object):

    def __init__(
            self,
            n_layers=10,
            embeddings_file={'train': 'input/emb10_64_cased_train.json', 'test': 'input/emb10_64_cased_test.json'}):
        self.layer_indexes = [-(l + 1) for l in range(n_layers)]
        self.embeddings_file = embeddings_file
        self.buckets = [1, 2, 3, 4, 5, 8, 16, 32, 64]
        self.init_seed(1122)

    def init_seed(self, seed):
        self.seed = seed
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed)
        rn.seed(seed)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(seed)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

    def load_embeddings(self, filename, data, idx_map, idx_samples):

        with open(filename) as f:
            for line in tqdm(f):
                sample = json.loads(line)
                if sample['segment'] > 0:
                    continue
                layers = []
                for layer in sample['embeddings']:
                    layers += layer['values']
                idx = sample['idx']
                idx_map[idx] = len(data['emb'])
                df_idx = sample['df_idx']
                if df_idx not in idx_samples:
                    idx_samples[df_idx] = []
                idx_samples[df_idx].append(len(data['emb']))
                # y = [0, 0, 0]
                # y[sample['label']] = 1
                # data[name]['y'].append(y)
                data['ids'].append(idx)
                data['df_ids'].append(df_idx)
                layers = np.array(layers)
                data['emb'].append([layers, np.zeros(layers.shape), np.zeros(layers.shape)])

        with open(filename) as f:
            for line in tqdm(f):
                sample = json.loads(line)
                if sample['segment'] == 0:
                    continue
                layers = []
                for layer in sample['embeddings']:
                    layers += layer['values']
                idx = sample['idx']
                segment = sample['segment']
                data['emb'][idx_map[idx]][segment] = np.array(layers)

        return data, idx_map, idx_samples

    def load_feats(self, filename, ids_filename, data, idx_samples):
        feature_idx_map = []
        with open(ids_filename) as f:
            f.readline()
            for line in f:
                feature_idx_map.append(line.strip().split('\t')[0])

        with open(filename) as f:
            f.readline()
            for i, line in enumerate(f):
                arr = line.strip().split('\t')
                feats = [float(x) for x in arr[1:]]
                idx = feature_idx_map[int(arr[0])]
                for sample_id in idx_samples[idx]:
                    data[sample_id] = feats

        return data

    def load_labels(self, filename, data, idx_samples):
        df = pd.read_csv(filename, sep='\t')
        df['label'] = 0
        if 'A-coref' in df.columns:
            df.loc[df['A-coref'] == True, 'label'] = 1
            df.loc[df['B-coref'] == True, 'label'] = 2
        for i, row in df.iterrows():
            idx = row.ID
            for sample_id in idx_samples[idx]:
                data[sample_id][row['label']] = 1
        return data

    def get_model(self, input_shapes):
        feat_shape = input_shapes[0]
        emb_shapes = input_shapes[1:]

        def build_emb_model(input_shape, emb_num=0, dense_layer_size=128, dropout_rate=0.9):
            X_input = layers.Input([input_shape])
            X = layers.Dense(dense_layer_size, name='emb_dense_{}'.format(emb_num),
                             kernel_initializer=initializers.glorot_uniform(seed=self.seed))(X_input)
            X = layers.BatchNormalization(name='emb_bn_{}'.format(emb_num))(X)
            X = layers.Activation('relu')(X)

            X = layers.Dropout(dropout_rate, seed=self.seed)(X)

            # Create model
            model = models.Model(inputs=X_input, outputs=X, name='emb_model_{}'.format(emb_num))
            return model

        emb_models = []
        for i, emb_shape in enumerate(emb_shapes):
            triplet_input = layers.Input([emb_shape])
            triplet_model_shape = int(emb_shape / 3)
            triplet_model = build_emb_model(triplet_model_shape, emb_num=i, dense_layer_size=112)
            P = layers.Lambda(lambda x: x[:, :triplet_model_shape])(triplet_input)
            A = layers.Lambda(lambda x: x[:, triplet_model_shape: triplet_model_shape * 2])(triplet_input)
            B = layers.Lambda(lambda x: x[:, triplet_model_shape * 2: triplet_model_shape * 3])(triplet_input)
            A_out = triplet_model(layers.Subtract()([P, A]))
            B_out = triplet_model(layers.Subtract()([P, B]))
            triplet_out = layers.concatenate([A_out, B_out], axis=-1)
            merged_model = models.Model(inputs=triplet_input, outputs=triplet_out, name='triplet_model_{}'.format(i))
            emb_models.append(merged_model)

        emb_num = len(emb_models)

        emb_models += [build_emb_model(emb_shape, emb_num=i + emb_num, dense_layer_size=112) for i, emb_shape in
                       enumerate(emb_shapes)]

        def build_feat_model(input_shape, dense_layer_size=128, dropout_rate=0.8):
            X_input = layers.Input([input_shape])
            X = layers.Dense(dense_layer_size, name='feat_dense_{}'.format(0),
                             kernel_initializer=initializers.glorot_normal(seed=self.seed))(X_input)
            X = layers.Activation('relu')(X)
            X = layers.Dropout(dropout_rate, seed=self.seed)(X)

            # Create model
            model = models.Model(inputs=X_input, outputs=X, name='feat_model')
            return model

        feat_model = build_feat_model(feat_shape, dense_layer_size=128)

        lambd = 0.02  # L2 regularization
        # Combine all models into one model

        merged_out = layers.concatenate([feat_model.output] + [emb_model.output for emb_model in emb_models])
        merged_out = layers.Dense(3, name='merged_output', kernel_regularizer=regularizers.l2(lambd),
                                  kernel_initializer=initializers.glorot_uniform(seed=self.seed))(merged_out)
        merged_out = layers.BatchNormalization(name='merged_bn')(merged_out)
        merged_out = layers.Activation('softmax')(merged_out)
        combined_model = models.Model([feat_model.input] + [emb_model.input for emb_model in emb_models],
                                      outputs=merged_out, name='merged_model')
        # print(combined_model.summary())

        return combined_model

    def train_model(self, embeddings, features, input_files):
        learning_rate = 0.02
        decay = 0.03
        n_fold = 5
        batch_size = 64
        epochs = 10000
        patience = 50
        # n_test = 100

        test_ids = {}
        test_ids_list = []
        Y_test = []
        for model_i, embeddings_files in enumerate(embeddings):

            for name in ['test', 'train']:
                print('Processing {} datasets'.format(name))

                for file_i, embeddings_file in enumerate(embeddings_files[name]):
                    if file_i == 0:
                        data, idx_map, idx_samples = self.load_embeddings(
                            embeddings_file,
                            data={'emb': [], 'ids': [], 'df_ids': []},
                            idx_map={},
                            idx_samples={})
                    else:
                        data, idx_map, idx_samples = self.load_embeddings(embeddings_file, data, idx_map, idx_samples)

                for i, emb in enumerate(data['emb']):
                    data['emb'][i] = np.concatenate(emb)

                if model_i == 0:
                    feats = [[] for _ in range(len(data['emb']))]
                    labels = [[0, 0, 0] for _ in range(len(data['emb']))]

                    if name == 'train':
                        X_emb_train = [np.array(data['emb'])]
                    else:
                        X_emb_test = [np.array(data['emb'])]

                    # Load features
                    for features_i, filename in enumerate(features[name]):
                        feats = self.load_feats(filename, features['{}_ids'.format(name)][features_i], feats,
                                                idx_samples)

                    if name == 'train':
                        X_feats_train = np.array(feats)
                    else:
                        X_feats_test = np.array(feats)

                    # Load labels
                    for filename in input_files[name]:
                        labels = self.load_labels(filename, labels, idx_samples)

                    if name == 'train':
                        Y_train = np.array(labels)
                    else:
                        for data_i, idx in enumerate(data['df_ids']):
                            if idx not in test_ids:
                                test_ids_list.append(idx)
                                test_ids[idx] = len(test_ids)
                                Y_test.append(labels[data_i])
                        Y_test = np.array(Y_test)
                else:
                    if name == 'train':
                        X_emb_train.append(np.array(data['emb']))
                    else:
                        X_emb_test.append(np.array(data['emb']))

        print('Train shape:', [x.shape for x in X_emb_train], X_feats_train.shape)
        print('Test shape:', [x.shape for x in X_emb_test], X_feats_train.shape)

        # Normalise feats
        need_normalisation = True
        if need_normalisation:
            all_feats = np.concatenate([X_feats_train, X_feats_test])
            all_max = np.max(all_feats, axis=0)
            X_feats_train /= all_max
            X_feats_test /= all_max

        model_shapes = [X_feats_train.shape[1]] + [x.shape[1] for x in X_emb_train]
        X_test = [X_feats_test] + X_emb_test + X_emb_test

        Y_test = np.array(Y_test)
        prediction = np.zeros((len(test_ids), 3))  # testing predictions
        prediction_cnt = np.zeros((len(test_ids), 3))  # testing predictions counts

        # for seed in [1, 6033, 100007]:
        for seed in [1122]:
            # for seed in [1, ]:
            # self.init_seed(seed)

            # Training and cross-validation
            # folds = GroupKFold(n_splits=n_fold)
            folds = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

            scores = []
            # for fold_n, (train_index, valid_index) in enumerate(folds.split(X_emb_train, groups=groups)):
            for fold_n, (train_index, valid_index) in enumerate(folds.split(Y_train)):
                # split training and validation data
                print('Fold', fold_n, 'started at', time.ctime())
                X_tr = [X_feats_train[train_index]] + [x[train_index] for x in X_emb_train] + [x[train_index] for x in
                                                                                               X_emb_train]
                X_val = [X_feats_train[valid_index]] + [x[valid_index] for x in X_emb_train] + [x[valid_index] for x in
                                                                                                X_emb_train]
                Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]

                # Define the model, re-initializing for each fold
                classif_model = self.get_model(model_shapes)
                classif_model.compile(optimizer=optimizers.Adam(lr=learning_rate, decay=decay),
                                      loss="categorical_crossentropy")
                callbacks = [kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]

                # train the model
                classif_model.fit(x=X_tr, y=Y_tr, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                                  validation_data=(X_val, Y_val), verbose=0)

                # make predictions on validation and test data
                pred_valid = classif_model.predict(x=X_val, verbose=0)
                pred = classif_model.predict(x=X_test, verbose=0)

                print('Stopped at {}, score {}'.format(callbacks[0].stopped_epoch, log_loss(Y_val, pred_valid)))

                scores.append(log_loss(Y_val, pred_valid))
                for i, idx in enumerate(test_ids_list):
                    prediction[test_ids[idx]] += pred[i]
                    prediction_cnt[test_ids[idx]] += np.ones(3)

            # Print CV scores, as well as score on the test data
            print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
            print(scores)
            print("Test score (original dirty labels):", log_loss(Y_test, prediction / prediction_cnt))

        prediction /= prediction_cnt

        # Write the prediction to file for submission
        np.savetxt('test_prediction.txt', prediction)
