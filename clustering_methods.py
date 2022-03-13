# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


def kmeans(dataset, n_clusters, random_state=None):

    km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(dataset)
    return km


def kmedoids(dataset, n_clusters, random_state=None):

    kmd = KMedoids(n_clusters=n_clusters, random_state=random_state).fit(dataset)
    return kmd