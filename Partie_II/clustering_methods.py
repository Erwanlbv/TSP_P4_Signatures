# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans


def kmeans(dataset, n_clusters):
    return KMeans(n_clusters).fit(dataset)


