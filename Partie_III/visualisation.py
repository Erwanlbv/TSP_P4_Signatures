# -*- conding: utf-8 -*-

import numpy as np

from variables import *
from dataset_prepro import get_eval_dataset
from KMclass import KMclassificator
from classificateurs import decision_tree, random_fo

COMPLETE_DATASET = fich_numpy_24


def test_simple():

    # On utilise un état initial fixe pour pouvoir comparer avec l'entrainement sur les 2500 complexités
    km_wrapper = KMclassificator(dataset=COMPLETE_DATASET, random_state=0)

    km_wrapper.train(method='1250') # 1250, half ou random

    # ---- Initilisation des bases de données ----

    x_train = km_wrapper.first_1250_dataset # Base de données de longueur 1250 pour l'entrainement
    x_train_labels = km_wrapper.km.labels_ # Les étiquettes associées obtenues par Kmeans

    x_eval = get_eval_dataset(complete_dataset=COMPLETE_DATASET, train_dataset=x_train)


    # ---- Entrainement des classificateurs ----

    tree_clf = decision_tree(x_train=x_train, x_labels=x_train_labels)
    rand_fo_clf = random_fo(x_train=x_train, x_labels=x_train_labels)


    # ---- Prédictions ----

    tree_labels = tree_clf.predict(x_eval)
    rand_fo_clf_labels = rand_fo_clf.predict(x_eval)
    kmeans_1250_labels = km_wrapper.km.predict(x_eval)

    # Et pour un entrainement avec 2500 signatures :
    kmeans_all = km_wrapper.train()
    kmeans_all_labels = km_wrapper.km.labels_


    # ---- Résultats, affichage et interprétation ----













