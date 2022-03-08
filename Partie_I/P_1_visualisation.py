import matplotlib.pyplot as plt

from clustering_methods import kmeans_clust, kmedoids_clust, spectral_clust
from scores import compute_scores
from variables import *


fich_4 = np.copy(fich_numpy_4)
fich_8 = np.copy(fich_numpy_8)
fich_24 = np.copy(fich_numpy_24)

datas = [fich_numpy_4, fich_numpy_8, fich_numpy_24]
nb_gauss = [4, 8, 24]

MIN_CLUSTERS = 2
MAX_CLUSTERS = 20


def kmeans_visualisation():

    for id, n_gauss in enumerate(nb_gauss):
        data = datas[id] #Matrice 100*25
        sign_moy_vect = data.mean(axis=1) # Vecteur 100*1
        prep_for_sklearn = sign_moy_vect.reshape(len(sign_moy_vect), -1)

        #iner_bool, sil_bool, davies_bool = True, True, True
        #scores = np.zeros((MAX_CLUSTERS + 1 - MIN_CLUSTERS, 3))
        #
        #for i in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        #    km = kmeans_clust(numpy_moy_for_sklearn, i)
        #    scores[i - MIN_CLUSTERS] = compute_scores(numpy_moy_for_sklearn, km, iner_bool, sil_bool, davies_bool)
        #display_res('Kmeans Clustering', scores)

        km = kmeans_clust(prep_for_sklearn, 3)
        display_classification(km, sign_moy_vect, n_gauss)


def kmedoids_visualization():

    for nb in nb_gauss:
        numpy_moy_for_sklearn = 1, 2
        iner_bool, sil_bool, davies_bool = True, True, True
        scores = np.zeros((MAX_CLUSTERS + 1 - MIN_CLUSTERS, 3))

        for i in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            kmedoids = kmedoids_clust(numpy_moy_for_sklearn, i)
            scores[i - MIN_CLUSTERS] = compute_scores(numpy_moy_for_sklearn, kmedoids, iner_bool, sil_bool, davies_bool)

        display_res('Kmedoids Clustering', scores)


def spectral_clustering():

    for nb in nb_gauss:
        numpy_moy_for_sklearn = 1, 2
        iner_bool, sil_bool, davies_bool = False, True, True # Pas d'inertie pour un regroupement spectral
        scores = np.zeros((MAX_CLUSTERS + 1 - MIN_CLUSTERS, 3))

        for i in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            spect_clus = spectral_clust(numpy_moy_for_sklearn, i)
            scores[i - MIN_CLUSTERS] = compute_scores(numpy_moy_for_sklearn, spect_clus, iner_bool, sil_bool, davies_bool)

        display_res('Spectral Clustering', scores)


def display_res(method_name, res):

    fig, axs = plt.subplots(3, figsize=(13, 7))
    fig.suptitle("Méthode de regroupement : " + method_name, fontsize=12)

    for id, ax in enumerate(axs.flat):
        ax.set_title("Inertie du modèle : " * (id == 0) + 'Score Silouhette : ' * (id == 1) + 'Davies Score' * (id == 2))
        ax.plot(range(MIN_CLUSTERS, MAX_CLUSTERS + 1), res[:, id])
        if id < 2:
            ax.get_xaxis().set_visible(False)

    fig.show()


def display_signatures():

    sign_4 = fich_4[:10]
    sign_8 = fich_8[:10]
    sign_24 = fich_24[:10]

    L_sign = [sign_4, sign_8, sign_24]

    fig, axs = plt.subplots(3, figsize=(13, 10))
    fig.suptitle('Moyenne et Écart type de complexité sur 5 signatures pour chaque mélange', fontsize=15)

    for id, sign in enumerate(L_sign):

        nb_gauss = (id == 0) * '4' + (id == 1) * '8' + (id == 24) * '24'
        axs[id].set_title('Pour ' + nb_gauss + ' gausiennes')
        axs[id].boxplot(sign.T, widths=0.2)
        axs[id].yaxis.grid(True)
        axs[id].set_xlabel(None)

        axs[id].set_ylim(16, 34)


    fig.show()


def display_classification(kmeans, data, n_gauss):

    fig, ax = plt.subplots()
    fig.suptitle('Classification des signatures pour ' + str(n_gauss) + ' gaussiennes')
    ax.scatter(range(100), data, c=kmeans.labels_)
    fig.show()


#kmeans_visualisation()
#kmedoids_visualization()
#spectral_clustering()
display_signatures()











