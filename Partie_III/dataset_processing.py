# -*- conding: utf-8 -*-

import numpy as np

from variables import *
from dataset_gen import firsts_dataset, half_dataset, random_dataset
from clustering_methods import kmeans

COMPLETE_DATASET = fich_numpy_24.flatten()


class Bdd():

    def __init__(self, method=None):

        if method == 'firsts':
            self.data_train, self.data_eval = firsts_dataset(dataset=COMPLETE_DATASET)
        elif method == 'half':
            self.data_train, self.data_eval = half_dataset(dataset=COMPLETE_DATASET)
        elif method == 'random':
            self.data_train, self.data_eval = random_dataset(dataset=COMPLETE_DATASET)

        else:
            print('nom de méthode donnée incorrect')

        # On adapte les bases de données à Kmeans
        self.data_train = self.data_train.reshape(len(self.data_train), -1)
        self.data_eval = self.data_eval.reshape(len(self.data_eval), -1)

        self.km = kmeans(self.data_train, n_clusters=3, random_state=0)

        # Random_state est utilisé ici pour pouvoir comparer des kmeans
        # entrainés sur d'autres bases de données en conservant le même état initial.

        self.km_data_train_labels = self.km.labels_
        self.km_data_eval_labels = self.km.predict(self.data_eval)

    def get_train_bdd(self):
        return self.data_train, self.km_data_train_labels

    def get_eval_bdd(self):
        return self.data_eval, self.km_data_eval_labels

    def get_km(self):
        return self.km