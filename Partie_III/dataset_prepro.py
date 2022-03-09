# -*- coding: utf-8 -*-

import numpy as np

# Les bases de données doivent avoir le format  (2500, )


def first_1250_dataset(dataset):

    new_dataset = dataset[:1250]
    return new_dataset


def random_dataset(dataset):

    indices = 1250 * np.random.rand(1250)
    indices = indices.astype(int)

    new_dataset = dataset[indices]
    return new_dataset


def first_half(dataset):

    dataset = dataset.reshape(100, 25)
    new_dataset = dataset[0, :13]
    for i in range(1, 100):
        new_dataset = np.concatenate((new_dataset, dataset[i, :(13 * (i < 50) + 12 * (i >= 50))]))

    return new_dataset


