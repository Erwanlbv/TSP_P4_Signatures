# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def decision_tree(x_train, x_label):

    max_depth = None # Profondeur de l'arbre
    min_samples_split = 2 # Nombre minimal de données pour pouvoir séparer en 2 branches

    return DecisionTreeClassifier(max_depth=max_depth,
                                  min_samples_split=min_samples_split).fit(x_train, x_label)


def random_fo(x_train, x_label):
    n_arbres = 10
    max_depth = None
    min_samples_split = 2

    return RandomForestClassifier(n_estimators=n_arbres,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split).fit(x_train, x_label)











