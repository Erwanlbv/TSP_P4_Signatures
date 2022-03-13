import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score


def compute_scores(dataset, km, inertia_bool=False, sil_bool=False, davies_bool=False):
    output = [None, None, None]

    if inertia_bool:
        output[0] = np.round(km.inertia_, 3)

    if sil_bool: # On cherche à la maximiser, prends des valeurs entre -1 et 1
        output[1] = np.round(silhouette_score(dataset, km.labels_), 3)

    if davies_bool: # On cherche à le minimiser (la valeur minimale étant 0)
        output[2] = np.round(davies_bouldin_score(dataset, km.labels_), 3)

    return output