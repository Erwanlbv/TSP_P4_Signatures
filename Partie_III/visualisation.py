# -*- conding: utf-8 -*-

import numpy as np

from variables import *
from dataset_gen import first_1250_dataset, first_half, random_dataset
from clustering_methods import kmeans


COMPLETE_DATASET = fich_numpy_24


# On obtient les 3 bases de données (qui serviront à comparer les entrainements)
_1250_data_train, _1250_data_eval = first_1250_dataset(dataset=COMPLETE_DATASET)
_half_data_train, _half_data_eval = first_half(dataset=COMPLETE_DATASET)
_random_data_train, _random_data_eval = random_dataset(dataset=COMPLETE_DATASET)


km_half = kmeans(_half_data_train, n_clusters=3, random_state=0)
km_random = kmeans(_random_data_train, n_clusters=3, random_state=0)


def new_random_entities():
    _random_data_train, _random_data_eval = random_dataset(dataset=COMPLETE)
    km_random = kmeans(_random_data_train, n_clusters=3)


def display_results(clf):














