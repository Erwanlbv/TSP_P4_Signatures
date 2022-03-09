# -*- coding: utf-8 -*-

import numpy as np

# Les bases de données doivent avoir le format  (2500, )


def first_1250_dataset(dataset):

    new_dataset = dataset[:1250]
    eval_dataset = dataset[1250:]
    return new_dataset, eval_dataset


def random_dataset(dataset):

    indices = 1250 * np.random.rand(1250)
    indices = indices.astype(int)

    new_dataset = dataset[indices]
    eval_dataset = get_eval_dataset(dataset, new_dataset)
    return new_dataset, eval_dataset


def first_half(dataset):

    dataset = dataset.reshape(100, 25)
    new_dataset = dataset[0, :13]
    for i in range(1, 100):
        new_dataset = np.concatenate((new_dataset, dataset[i, :(13 * (i < 50) + 12 * (i >= 50))]))

    eval_dataset = get_eval_dataset(dataset, new_dataset)
    return new_dataset, eval_dataset


def get_eval_dataset(complete_dataset, train_dataset):

    # CETTE METHODE FONCTIONNE UNIQUEMENT PARCE QUE TOUS LES ELEMENTS DE LA BASE DE DONNEES SONT DIFFERENTS
    # Vérification faite avec np.sum(np.unique(dataset.flatten(), return_counts=True)[1]) = 2500
    # soit le nombre d'éléments de complete_dataset.

    indices = []
    for i, complexity in enumerate(complete_dataset):
        if not complexity in train_dataset:
             indices.append(i)

    return indices




