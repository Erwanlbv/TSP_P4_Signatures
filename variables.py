import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Signature avec 4 gaussiennes :
fich_4 = pd.read_csv('/Users/erwan/PycharmProjects/TSP_P4_Signatures/Donnees_Moodle/Complexite_avec_4G.txt', sep='\t', header=None)
fich_4.astype(float)
fich_numpy_4 = fich_4.to_numpy() # Fichier des signatures au format Numpy

fich_moy_4 = fich_numpy_4.mean(axis=1) # Moyennes au format Dataframe

# Signature avec 8 gaussiennes :
fich_8 = pd.read_csv('/Users/erwan/PycharmProjects/TSP_P4_Signatures/Donnees_Moodle/Complexite_avec_8G.txt', sep='\t', header=None)
fich_8.astype(float)
fich_numpy_8 = fich_8.to_numpy() # Fichier des signatures au format Numpy

fich_moy_8 = fich_numpy_8.mean(axis=1) # Moyennes au format Dataframe

# Signature avec 24 gaussiennes :
fich_24 = pd.read_csv('/Users/erwan/PycharmProjects/TSP_P4_Signatures/Donnees_Moodle/Complexite_avec_24G.txt', sep='\t', header=None)
fich_24.astype(float)
fich_numpy_24 = fich_24.to_numpy() # Fichier des signatures au format Numpy

fich_moy_24 = fich_24.mean(axis=1) # Moyennes au format Dataframe

# ZONES DES 2500 SIGNATURES

#dataframe = pd.read_csv('/Donnees_Moodle/Base_de_donnees.rar')
