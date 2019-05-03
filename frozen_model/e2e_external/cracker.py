import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.io.json import json_normalize
import os
import gc
import time
import re

np.random.seed(77)

class Cracker :

    def __init__(self, name, data_folder = ".", output_folder = ".") :
        print('cracker constructor ', name)
        self.name = name
        # self.model_folder = os.path.join('.', name)
        #
        # print('model folder = ', self.model_folder)
        # if not os.path.exists(self.model_folder) :
        #     os.makedirs(self.model_folder)

        # if kernel :
        #     folder = '../input/petfinder-adoption-prediction' if images else '../input'
        # else :
        #     folder = '/storage/bigdata/Logvinenko/Pets'

        #output_folder = "" if kernel else folder


        self.data_folder = data_folder
        self.output_folder = output_folder

    # Переопределяем в наследниках
    def init_data(self) :
        pass

    # потом можно дополнить для валидации
    def save_data(self):
        self.train.to_pickle(os.path.join(self.output_folder, 'train.pkl'))
        self.test.to_pickle(os.path.join(self.output_folder, 'test.pkl'))

    # подготовка всех фичей и старых и новых
    def make_features(self, df):
        pass

    # при необходимости дополняем для трейна
    def prepare_train(self):
        pass

    def prepare_data(self):
        self.make_features(self.train)
        self.make_features(self.test)

        self.prepare_train()

        self.save_data()

    # готовим новые фичи и сохраняем
    def prepare_new_data(self):
        self.make_new_features(self.train)
        self.make_new_features(self.test)

        self.save_data()

    # загружаем сохраненные датасеты
    def load_data(self):
        self.train = pd.read_pickle(os.path.join(self.output_folder, 'train.pkl'))
        self.test = pd.read_pickle(os.path.join(self.output_folder, 'test.pkl'))

    def add_np(self, X_train, X_test):
        return (np.concatenate((self.train.values, X_train), axis = 1),
                np.concatenate((self.test.values, X_test), axis = 1))