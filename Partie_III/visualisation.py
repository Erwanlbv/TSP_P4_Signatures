# -*- conding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from clf_processing import train_clf, predict
from dataset_processing import Bdd
from labels_processing import get_partitions

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def test_simple():

    firsts_bdd = Bdd('firsts')
    train_data, train_labels = firsts_bdd.get_train_bdd()
    eval_data, eval_labels = firsts_bdd.get_eval_bdd()

    clfs = train_clf(train_data, train_labels) # tree, rando, total_km

    tree_eval_labels = clfs[1].predict(eval_data) # UTILISER CROSS VALIDE PLUTOT

    precision = accuracy_score(eval_labels, tree_eval_labels)
    cross_val = cross_val_score(clfs[1], eval_data, eval_labels)

    train_partition = get_partitions(clfs[1].predict(train_data))
    eval_partition = get_partitions(tree_eval_labels)

    print('Précision', precision)
    print('Score validation croisée', cross_val)
    print("Partition sur la base de données d'entrainement", train_partition)


test_simple()





