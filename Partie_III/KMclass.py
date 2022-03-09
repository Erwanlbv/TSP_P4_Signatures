# -*- coding: utf-8 -*-


from clustering_methods import kmeans
from Partie_III.dataset_prepro import first_1250_dataset, first_half, random_dataset


class KMclassificator():

    n_clusters = 3

    def __init__(self, dataset, random_state=None):

        self.dataset = dataset.flatten()

        self.first_1250_dataset = first_1250_dataset(self.dataset)
        self.first_half_dataset = first_half(self.dataset)

        self.random_dataset_initialized = False
        self.random_dataset = None

        self.km = None
        self.random_state = random_state

    def random_dat(self):
        dataset = random_dataset(self.dataset)

        self.random_dataset_initialized = True
        self.random_dataset = dataset

    def train(self, method=''): # 'method' doit être un string

        if method == "1250":
            dataset = self.first_1250_dataset.reshape(len(self.first_1250_dataset), -1) # On change le format pour kmeans
            self.km = kmeans(dataset=dataset, n_clusters=self.n_clusters, random_state=self.random_dataset)

        elif method == 'half':

            dataset = self.first_half_dataset.reshape(len(self.first_half_dataset), -1) # Idem
            self.km = kmeans(dataset=dataset, n_clusters=self.n_clusters, random_state=self.random_dataset)

        elif method == 'random':

            if not self.random_dataset_initialized:
                print("La base de données aléatoire n'a pas été générée, veuillez utiliser .random_init() avant "
                      "d'appeler cette méthode")

            else:
                dataset = self.random_dataset.reshape(len(self.random_dataset), -1) # Idem
                self.km = kmeans(dataset=dataset, n_clusters=self.n_clusters, random_state=self.random_dataset)

        else:
            print('Entrainement sur la totalité de la base de données')
            dataset = self.dataset.reshape(len(self.dataset), -1)
            self.km = kmeans(dataset, n_clusters=self.n_clusters, random_state=self.random_dataset)









