import matplotlib.pyplot as plt

from clustering_methods import kmeans, kmedoids
from scores import compute_scores
from variables import *


fich_4 = np.copy(fich_numpy_4)
fich_8 = np.copy(fich_numpy_8)
fich_24 = np.copy(fich_numpy_24)

datas = [fich_4, fich_8, fich_24]
nb_gauss = [4, 8, 24]

MIN_CLUSTERS = 2
MAX_CLUSTERS = 20


def kmeans_visualisation():

    for id, n_gauss in enumerate(nb_gauss):
        data = datas[id] #Matrice 100*25
        sign_moy_vect = data.mean(axis=1) # Vecteur 100*1
        data_for_sklearn = sign_moy_vect.reshape(len(sign_moy_vect), -1)

        km = kmeans(data_for_sklearn, 3)
        if n_gauss == 24:
            print(km.cluster_centers_)

        display_classification(km, sign_moy_vect, n_gauss)


def kmedoids_visualization():

    for id, n_gauss in enumerate(nb_gauss):
        data = datas[id] #Matrice 100*25
        sign_moy_vect = data.mean(axis=1) # Vecteur 100*1
        data_for_sklearn = sign_moy_vect.reshape(len(sign_moy_vect), -1)

        km = kmedoids(data_for_sklearn, 3)
        if n_gauss == 24:
            print(km.cluster_centers_)

        display_classification(km, sign_moy_vect, n_gauss)


def spectral_clustering():

    for nb in nb_gauss:
        numpy_moy_for_sklearn = 1, 2
        iner_bool, sil_bool, davies_bool = False, True, True # Pas d'inertie pour un regroupement spectral
        scores = np.zeros((MAX_CLUSTERS + 1 - MIN_CLUSTERS, 3))

        for i in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
            pass
            #spect_clus = spectral_clust(numpy_moy_for_sklearn, i)
            #scores[i - MIN_CLUSTERS] = compute_scores(numpy_moy_for_sklearn, spect_clus, iner_bool, sil_bool, davies_bool)

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

    for id, sign in enumerate(L_sign):

        fig, ax = plt.subplots(figsize=(13, 10))
        fig.suptitle('Moyenne et Écart type de complexité sur 5 signatures tirées aléatoirement', fontsize=15)

        nb_gauss = (id == 0) * '4' + (id == 1) * '8' + (id == 2) * '24'
        ax.set_title('Pour ' + nb_gauss + ' gausiennes')
        ax.boxplot(sign.T, widths=0.2)
        ax.yaxis.grid(True)
        ax.set_xlabel(None)

        ax.set_ylim(16, 34)


        fig.show()


def display_classification(kmeans, data, n_gauss):

    fig, ax = plt.subplots()
    fig.suptitle('Classification des signatures pour ' + str(n_gauss) + ' gaussiennes')
    ax.scatter(range(100), data, c=kmeans.labels_)
    ax.yaxis.grid(True)
    ax.set_xlabel(None)
    fig.show()


def display_mean_complexity_var():

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle('Evolution de la complexité moyenne des signatures en fonction du nombre de gaussiennes')

    x = range(100)
    ax.plot(x, fich_moy_4, label='4 gaussiennes', alpha=0.7)
    ax.plot(x, fich_moy_8, label='8 gaussiennes',  alpha=0.7)
    ax.plot(x, fich_moy_24, label='24 gaussiennes', alpha=0.7)

    ax.legend()
    fig.show()


kmeans_visualisation()
kmedoids_visualization()
#display_signatures()
#display_mean_complexity_var()









