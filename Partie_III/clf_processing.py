# -*- conding: utf-8 -*-

import numpy as np

from variables import *
from clustering_methods import kmeans
from clf import decision_tree, random_fo

COMPLETE_DATASET = np.copy(fich_24)


def train_clf(data_train, data_labels):
    tree_clf = decision_tree(data_train, data_labels)
    rand_fo_clf = random_fo(data_train, data_labels)

    # Pour le lien avec la partie II
    total_km_clf = kmeans(COMPLETE_DATASET, n_clusters=3)

    return tree_clf, rand_fo_clf, total_km_clf


def predict(clfs, data_eval):
    tree_clf, rand_fo_clf, total_km_clf = clfs

    tree_preds = tree_clf.predict(data_eval)
    rand_fo_preds = rand_fo_clf.predict(data_eval)

    # Pour le lien avec la partie II :
    total_km_preds = total_km_clf.labels_

    return tree_preds, rand_fo_preds, total_km_preds