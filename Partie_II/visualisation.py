# -*- coding: utf-8 -*-

from variables import *
from clustering_methods import kmeans

fich_24 = np.copy(fich_moy_24)

datas = [fich_24]

N_CLUSTERS = 3


def multiple_kmeans_class(dataset):

    nb_kmeans = 5
    labels_res = []
    for i in range(nb_kmeans):
        km = kmeans(dataset, n_clusters=N_CLUSTERS)
        labels_res.append(km.labels_)

    print(len(labels_res))
    print(labels_res[0].shape)


multiple_kmeans_class(datas[0])