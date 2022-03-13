# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def decision_tree(x_train, x_labels):
    max_depth = None  # Profondeur de l'arbre
    min_samples_split = 2  # Nombre minimal de données pour pouvoir séparer un paquet en 2 branches

    return DecisionTreeClassifier(max_depth=max_depth,
                                  min_samples_split=min_samples_split).fit(x_train, x_labels)


def random_fo(x_train, x_labels):
    n_arbres = 10
    max_depth = None
    min_samples_split = 2

    return RandomForestClassifier(n_estimators=n_arbres,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split).fit(x_train, x_labels)


