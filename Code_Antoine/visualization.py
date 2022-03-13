import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import time


#Lire les fichiers avec Pandas
df1 = pd.read_csv(
    'Complexite_avec_4G.txt', sep="\t",header=None)
df1 = df1.transpose()

df2 = pd.read_csv(
    'Complexite_avec_8G.txt', sep="\t",header=None)
df2 = df2.transpose()

df3 = pd.read_csv(
    'Complexite_avec_24G.txt', sep="\t",header=None)
df3 = df3.transpose()

##Faire une liste de listes pour les différentes signatures
Df = []
for i in range(0, 100):
    A = []
    for j in range(0, 25):
        A.append(pd.read_csv(
            'Base de données/' + str(i) + 'v' + str(j) + '.txt', sep=" ", header=None))
    Df.append(A)

######## Différents tests unitaires
# print(type(Df))
# print(len(Df))
# print(Df[11][12])
# print(len(Df[11][12]))
# print("ok")
# print(Df[11][12][0][1])

#---------------Description
# Df : liste de listes. 100 sous-listes, chacune de longueur 25 où chaque élément [j][i] représente la ième ligne et la jième colonne


###Extraire les coordonnées

#tracer_signature(7,12)

###Afficher toutes les signatures d'une personne
def tracer_signature_multiple(id_personne):
    # ax=[]
    # for i in range (0,25):
    #     ax.append(i)
    # print((ax[:]))
    fig,axs=plt.subplots(5,5)
    for i in range (0,5):
        a=5*i
        for j in range (0,5):
            b=a+j
            print(b)
            X, Y, Air = coordonnées_X_Y(id_personne, b)
            axs[i,j].plot(X, Y, color="black")
            axs[i,j].plot(X, Air, linestyle="--", linewidth="5", color="b")
            axs[i,j].set_title(str(i+j) + "ème signature de l'individu: " + str(id_personne))

            axs[i,j].set_ylabel("Coordonnée y")
            axs[i,j].set_xlabel("Coordonnée x")
    fig.show()
    plt.pause(10)

#tracer_signature_multiple(11)

# print(df1.shape)
# print(df1.shape[1])

######Moyenne des entropies différentielles

Df1=np.copy(df1)
Df2=np.copy(df2)
Df3=np.copy(df3)

df1=df1.mean(axis=0).to_numpy()
df2=df2.mean(axis=0).to_numpy()
df3=df3.mean(axis=0).to_numpy()

#print(type(df1), df2.shape,df3.shape)
# print(df1[0])

#########K-mean
kmeans1=KMeans(n_clusters=3,init='k-means++', random_state=1).fit(df1.reshape(-1,1))
kmeans2=KMeans(n_clusters=3,init='k-means++', random_state=1).fit(df2.reshape(-1,1))
kmeans3=KMeans(n_clusters=3,init='k-means++', random_state=2).fit(df3.reshape(-1,1))

# print(kmeans1.labels_)
# print(kmeans1.cluster_centers_)

####### % dans chaque cluster
def pourcentage(kmeans):
    P_0=0
    P_1=0
    P_2=0
    for i in range(0,len(kmeans.labels_)):
        if kmeans.labels_[i]==0:
            P_0+=1
        if kmeans.labels_[i]==1:
            P_1+=1
        if kmeans.labels_[i]==2:
            P_2+=1
    return P_0/len(kmeans.labels_),P_1/len(kmeans.labels_),P_2/len(kmeans.labels_)

print(pourcentage(kmeans1))

###Pie chart des différents clusters
def tracer_pourcentage(kmeans1,kmeans2,kmeans3):
    labels='Groupe simple', 'Groupe moyen', 'Groupe compliqué'
    fig,(ax1,ax2,ax3)=plt.subplots(1,3, figsize=(13,7))
    ax1.pie(pourcentage(kmeans1), labels=labels,autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.set_title("Pourcentage dans chaque cluster, \n avec un mélange à 4 gaussiennes")
    ax2.pie(pourcentage(kmeans2),labels=labels,autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax2.set_title("Pourcentage dans chaque cluster, \n avec un mélange à 8 gaussiennes")
    ax3.pie(pourcentage(kmeans3),labels=labels,autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax3.set_title("Pourcentage dans chaque cluster, \n avec un mélange à 24 gaussiennes")
    plt.show()

#tracer_pourcentage(kmeans1,kmeans2,kmeans3)


###Trouver les personnes dans chacun des clusters
def indice_classes(kmeans1):
    R_0=[]
    R_1=[]
    R_2=[]
    for i in range (0,len(kmeans1.labels_)):
        if kmeans1.labels_[i]==0:
            R_0.append(i)
        if kmeans1.labels_[i]==1:
            R_1.append(i)
        if kmeans1.labels_[i]==2:
            R_2.append(i)
    return R_0,R_1,R_2

###Tracer des représentants de chaque cluster
def tracer_représentant(kmeans):
    #R_0,R_1,R_2= indice_classes(kmeans)
    #R_0,R_1,R_2= indice_classes(kmeans)
    R_0,R_1,R_2= indice_classes(kmeans)

    fig,axs=plt.subplots(3,len(R_1))

    for i in range(0, len(R_1)):
        X, Y, Air = coordonnées_X_Y(R_1[i], 2)

        axs[0, i].plot(X, Y, color="black")
        axs[0, i].plot(X, Air, linestyle="--", linewidth="5", color="b")
        if i == 3:
            axs[0, i].set_title("Signature de la classe simple")
        axs[0, i].set_ylabel('Coordonnées y')
        #axs[0, i].set_xlabel("off")

    for i in range(0,len(R_1)):
        X,Y,Air=coordonnées_X_Y(R_0[i],2)

        axs[1, i].plot(X, Y, color="black")
        axs[1, i].plot(X, Air, linestyle="--", linewidth="5", color="b")
        if i == 3:
            axs[1, i].set_title("Signature de la classe moyenne")
        axs[1, i].set_ylabel('Coordonnées y')
        #axs[1, i].set_xlabel("off")


    for i in range(0, len(R_1)):
        X, Y, Air = coordonnées_X_Y(R_2[i], 2)

        axs[2, i].plot(X, Y, color="black")
        axs[2, i].plot(X, Air, linestyle="--", linewidth="5", color="b")
        if i == 3:
            axs[2, i].set_title("Signature de la classe complexe")
        axs[2, i].set_ylabel("Coordonnée y")
        axs[2, i].set_xlabel("Coordonnée x")

    plt.tight_layout()
    fig.show()
    plt.pause(50)

tracer_représentant(kmeans3)


####Partie 2
print(Df3.shape)
Entropie_matrix=[]
for i in range (0,100):
    for j in range (0,25):
        Entropie_matrix.append(Df3[j][i])
print(Entropie_matrix[0])
Entropie_matrix=np.array(Entropie_matrix)
Kmeans3=KMeans(n_clusters=3,init='k-means++', random_state=1).fit(Entropie_matrix.reshape(-1,1))
print(Kmeans3.labels_.shape)

def indice_matrix(Kmeans3):
    I=np.zeros((25,100))
    for i in range(0,100):
        for j in range(0,25):
            if Kmeans3.labels_[i*25+j] == 0:
                I[j][i]=0
            if Kmeans3.labels_[i*25+j] == 1:
                I[j][i] = 1
            if Kmeans3.labels_[i*25+j] == 2:
                I[j][i] = 2
    return I

I=indice_matrix(Kmeans3).transpose()
OK=[]
for i in range (0,100):
    if (I[i]==np.ones((1,25))).all():
        OK.append(i)

#print(OK)
# for i in OK:
#     print(I[i])

#tracer_signature_multiple(18)

####Seules les signatures très complexes sont toutes classées dans un même cluster

def proportion_attribution(Kmeans3,I):
    P=[]
    for i in range (0,100):
        P_0=0
        P_1=0
        P_2=0
        Tri_proportion=[]
        for j in range (0,25):
            if I[i][j]==0:
                P_0+=1
            if I[i][j]==1:
                P_1+=1
            if I[i][j]==2:
                P_2+=1
        Tri_proportion.append(P_0/25)
        Tri_proportion.append(P_1 / 25)
        Tri_proportion.append(P_2 / 25)
        P.append(Tri_proportion)
    return np.array(P)

P=proportion_attribution(Kmeans3,I)
print(P.shape)

def tracer_proportion(P):
    # bins=[0.001,0.2,0.4,0.6,0.8,1.001]
    # fig, ax= plt.subplots()
    # ax.hist(P[1],bins)
    plt.rcParams.update({'font.size': 4})
    labels= 'Groupe facile', 'Groupe complexe', 'Groupe moyen'
    fig,axs=plt.subplots(3,3)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(wspace=0, hspace=0.6)
    for i in range (0,3):
        for j in range (0,3):
            axs[i,j].pie(P[3*i+j+15],labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
            axs[i,j].set_title("Proportion des clusters de \n la" + str(3*i+j+15) + "-ème personne")
    fig.show()
    plt.pause(50)

#tracer_proportion(P)
#tracer_signature_multiple(7)