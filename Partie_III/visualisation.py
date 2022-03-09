# -*- conding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from clf_processing import train_clf, predict
from dataset_processing import Bdd


def test_simple():
    firsts_bdd = Bdd('firsts')
    train_data, train_labels = firsts_bdd.get_train_bdd()

    clfs = train_clf(train_data, train_labels) # tree, rando, total_km

    print(clfs)

test_simple()





