# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from variables import *
from clustering_methods import kmeans, kmedoids
from images_processing import get_displayable_coords, display_image

_4_moy = np.copy(fich_moy_4).reshape(len(fich_moy_4), -1)
_8_moy = np.copy(fich_moy_8).reshape(len(fich_moy_8), -1)
_24_moy = np.copy(fich_moy_24).reshape(len(fich_moy_24), -1)


def get_n_representants(n_rep, n_gauss):

    data = (n_gauss == 4) * _4_moy + (n_gauss == 8) * _8_moy + (n_gauss == 24) * _24_moy
    km = kmeans(dataset=data, n_cluster=3).fit(n_gauss)
    #km = kmedoids(dataset=data, n_clusters=3).fit(data)

    label_indices = []
    means = []
    for i in range(3):
        labels = np.where(km.labels_ == i)[0][:n_rep]
        mean = np.mean(data[labels])

        means.append(mean)
        means = np.sort(means).tolist()

        index = means.index(mean)
        label_indices.insert(index, labels)

    # Les labels sont maintenant triés pas complexité croissante dans label_indices, il ne reste
    # qu'à les afficher.

    fig, axs = plt.subplots(3, n_rep, figsize=(14, 9))
    fig.suptitle(str(n_rep) + ' centres de classes pour ' + str(n_gauss) + 'gaussiennes', fontsize=16)

    for i, indices in enumerate(label_indices):
        print('Format', indices.shape)
        for j in range(n_rep):
            signature = np.loadtxt('../Donnees_Moodle/Base_de_donnees/' + str(indices[j]) + 'v0.txt')[:, :3]
            sign_displayable_coords = list(get_displayable_coords(signature))

            axs[i, j].plot(sign_displayable_coords[0], sign_displayable_coords[1], color='black')
            axs[i, j].plot(sign_displayable_coords[0], sign_displayable_coords[2], linestyle='--', linewidth=2, color='b')
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)

    fig.show()


get_n_representants(5, 4)
get_n_representants(5, 8)
get_n_representants(5, 24)