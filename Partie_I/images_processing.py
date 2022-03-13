# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


num = 11
col = 8

#full_signature = np.loadtxt('../Donnees_Moodle/Base_de_donnees/' + str(num) + 'v' + str(col) + '.txt')
#signature = full_signature[:, :3]#
# print(signature.shape)


def get_one_person_signatures(id): # Récupère toutes les signatures (points x, y, p) d'une personne
    signatures = []
    for i in range(25):
        array = np.loadtxt('../Donnees_Moodle/Base_de_donnees/' + str(id) + 'v' + str(i) + '.txt')[:, :3]
        signatures.append(array)

    return np.array(signatures)


def get_displayable_coords(signature): # Pour permettre de représenter la signature en pointillés quand pression = 0

    y, air = np.nan * np.ones(len(signature)), np.nan * np.ones(len(signature))
    pression_indices, no_pression_indices = np.where(signature[:, 2] != 0), np.where(signature[:, 2] == 0)

    y[pression_indices] = signature[:, 1][pression_indices]
    air[no_pression_indices] = signature[:, 1][no_pression_indices]

    return signature[:, 0], y, air


def display_image(coords): # coords = [x, y, air], Affiche la signature associée aux coordonées.

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Signature n°" + str(col) + " de l'individu " + str(num), fontsize=12)
    ax.plot(coords[0], coords[1], color='black')
    ax.plot(coords[0], coords[2], linestyle='--', linewidth=2, color='b')
    ax.set_ylabel("Coordonnée y")
    ax.set_xlabel("Coordonnée x")
    fig.show()


def display_multiple_images(signatures): # Affiche toutes les signatures données en paramètre (dans une limite de 25)

    fig, axs = plt.subplots(5, 5, figsize=(13, 7))
    fig.suptitle("Signatures de l'individu n°" + str(num), fontsize=13)

    # On utilise axs.flat quitte à faire un flatten sur les signatures en entrée (plus pratique pour
    # appeler les axes, avec enumerate).

    for id, ax in enumerate(axs.flat):
        coords = get_displayable_coords(signatures[id])
        ax.plot(coords[0], coords[1], color='black', linewidth=1)
        ax.plot(coords[0], coords[2], linestyle='--', linewidth=2, color='b')

        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.show()


def get_one_person_complexity(id): # id est dans [0, 24], Retourne les complexités des signatures 4, 8 et 24G
    _4G_sign = np.loadtxt('../Donnees_Moodle/Complexite_avec_4G.txt')[id, :]
    _8G_sign = np.loadtxt('../Donnees_Moodle/Complexite_avec_8G.txt')[id, :]
    _24G_sign = np.loadtxt('../Donnees_Moodle/Complexite_avec_24G.txt')[id, :]

    return _4G_sign, _8G_sign, _24G_sign


def one_person_bougies(id): # id = Numéro de la personne (n° de ligne) dans la base de données

    signs = list(get_one_person_complexity(id))

    fig, ax = plt.subplots()
    fig.suptitle("Variations des complexités des signatures de \nl'individu " + str(id), fontsize=13)

    ax.boxplot(signs, widths=0.2)
    ax.yaxis.grid(True)
    ax.set_xlabel(None)

    fig.show()



#coords = get_displayable_coords(signature)
#display_image(coords)

#signatures = get_one_person_signatures(2)
#display_multiple_images(signatures)

#one_person_bougies(num)


