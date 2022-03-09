# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def get_labels(method, data):

    res = method.predict(data)
    return res


def get_partitions(labels):

    res = [0, 0, 0] # Nombre de personnes appartenant à 1 unique cluster, 2 clusters, 3 clusters

    for person_label in labels:
        n_clusters = len(np.unique(person_label))
        print('Personne apparetnant à ' + str(n_clusters) + ' clusters')

        res[n_clusters - 1] += 1

    return res


