# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans


def kmeans(dataset, n_clusters, random_state=None):

    km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(dataset)
    return km