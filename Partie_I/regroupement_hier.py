import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from variables import *

DATASET_24 = fich_numpy_24


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


datas = [np.copy(fich_moy_4).reshape(100, -1),
         np.copy(fich_moy_8).reshape(100, -1),
         np.copy(fich_moy_24).reshape(100, -1)
]


for id, data in enumerate(datas):

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)

    model = model.fit(data)
    fig = plt.figure(figsize=(12, 6))

    nb_gauss = (id == 0) * '4' + (id == 1) * '8' + (id == 2) * '24'
    plt.title("Hierarchical Clustering Dendrogram \n "
              "Pour " + str(nb_gauss) + ' gaussiennes')

    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Nombre de points dans le noeud (ou, si le point est seul, son index)")
    plt.show()