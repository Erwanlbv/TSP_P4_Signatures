# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from variables import *
from clustering_methods import kmeans


N_CLUSTERS = 3
DATASET = np.copy(fich_24) # Matrice 100*25
splited_data = np.split(DATASET, 5)

L = [i for i in range(100)]
M = np.array([[L[i]] * 25 for i in range(100)])
splited_M = np.array_split(M, 5)

print(type(splited_M))
print('Longueur', len(splited_M))
print('Format', splited_M[0].shape)


def display_mean_complexity():

    data = np.mean(DATASET, axis=1).reshape(100, -1)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Complexité moyenne des signatures pour 24 gaussiennes')

    ax.scatter(range(len(data)), data, c=range(100))

    ax.yaxis.grid(True)
    ax.set_xlabel(None)
    fig.show()


def get_separators(km):

    sep = []
    cluster_centers = np.sort(km.cluster_centers_.flatten()).tolist()
    prec_center = cluster_centers[0]

    for i in range(1, len(cluster_centers)):
        sep.append(((prec_center + cluster_centers[i]) / 2))
        prec_center = cluster_centers[i]

    print('Centres', cluster_centers)
    print('Séparateurs', sep)

    x_array = [0, 0, 2499, 2499]
    y_array = sep * 2

    return x_array, y_array


def display_classification(kmeans, data):

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Classification des signatures pour 24 gaussiennes')
    ax.scatter(range(len(data)), data, c=kmeans.labels_)
    ax.yaxis.grid(True)
    ax.set_xlabel(None)
    fig.show()


def kmeans_visualisation():
    l, c = DATASET.shape[0], DATASET.shape[1]
    data_for_sklearn = DATASET.reshape(l * c, -1) # Vecteur de longueur 2500

    km = kmeans(data_for_sklearn, 3, random_state=0)
    print(km.cluster_centers_)
    display_classification(km, data_for_sklearn)


def signs_visualisation(with_km=False):

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("Complexité de toutes les signatures d'après le mélange à 24 gaussiennes")

    for i in range(5):

        means = np.mean(splited_data[i], axis=1)
        index = np.linspace(i*500, (i+1) * 500, 20, dtype='int')

        ax.scatter(range(i*500, (i+1) * 500), splited_data[i], c=splited_M[i], cmap='tab20')
        ax.scatter(index, means, marker='+', s=75, color='black')

    if with_km:
        data = np.mean(DATASET, axis=1).reshape(100, -1)
        km = kmeans(data, n_clusters=3, random_state=0)

        km_global = kmeans(DATASET.reshape(2500, -1), n_clusters=3)

        x, y = get_separators(km)
        x_glob, y_glob = get_separators(km_global)

        ax.plot(x[::2], y[::2], linestyle=None, color='b', alpha=0.8, label='Pour les 100 moyennes')
        ax.plot(x[1::2], y[1::2], linestyle=None, color='b', alpha=0.8)
        ax.plot(x_glob[::2], y_glob[::2], linestyle=None, color='r', alpha=0.8, label='Pour les 2500 signatures')
        ax.plot(x_glob[1::2], y_glob[1::2], linestyle=None, color='r', alpha=0.8)

    ax.yaxis.grid(True)
    ax.set_xlabel(None)
    ax.legend()

    fig.show()


def display_25_person_complex(ids):

    fig, ax = plt.subplots()
    fig.suptitle("Représentation des 25 complexités de 13 individus")

    for id in ids:
        value = .9 + id/100
        person_complx = DATASET[id]
        ax.scatter([value]*25, person_complx, alpha=0.7)

    ax.set_ylabel('Complexité')
    ax.get_xaxis().set_visible(False)
    fig.show()


#display_mean_complexity()
#kmeans_visualisation()
#signs_visualisation()
#signs_visualisation(True)
display_25_person_complex([i for i in range(0, 25, 2)])