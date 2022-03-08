# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids


def kmeans_clust(dataset, n_clusters):

    km = KMeans(n_clusters=n_clusters, n_init=10).fit(dataset)
    return km


def spectral_clust(dataset, n_clusters):

    spec_cl = SpectralClustering(n_clusters=n_clusters).fit(dataset)
    return spec_cl


def kmedoids_clust(dataset, n_clusters):

    kmedoids = KMeans(n_clusters=n_clusters).fit(dataset)
    return kmedoids

