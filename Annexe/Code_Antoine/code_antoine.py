# Importer toutes les librairies utiles
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
#from sklearn_extra.cluster import KMedoids
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from skimage import color
from skimage import io
import numpy as np

#Importer le dataset et le segmenter en un ensemble d'entrainement (85%) et un ensemble de test (15%)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Vérifier les types
print("Check Type")
print(type(x_train))
print(type(x_test))
print(type(y_train))
print(type(y_test))
plt.imshow(x_train[0])


#On divise par dix le nombre d'images pour des soucis de temps de calcul

x_train=x_train[0:6000][:][:]
x_test=x_test[0:1000][:][:]
y_train=y_train[0:6000][:]
y_test=y_test[0:1000][:]

# Vérifier les dimensions
print("Check Type")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

###On veut des images en noir et blanc
plt.gray()
# Tracer 9 images parmi l'ensemble de test
fig, axs= plt.subplots(3,3)
for i in range (0,3):
    for j in range (0,3):
        axs[i,j].imshow(x_train[i+j])
        axs[i,j].set_title(str(i+j+1) + "-ème nombre écrit à la main")
        axs[i,j].set_xlabel("Abscisse : 28 pixels")
        axs[i, j].set_ylabel("Ordonné: 28 pixels")
        plt.subplots_adjust(hspace=0.5)
plt.show()
plt.close()

# Tracer 5 labels dans l'ensemble de test
for i in range(5):
  print("Label dans l'ensemble de test  "+str(y_train[i]))

# Obtenir la valeur min/max de l'ensemble de test
print("Valeur minimum de l'ensemble de test avant normalisation  " + str(x_train.min()))
print("Valeur minimum de l'ensemble de test avant normalisation  " + str(x_train.max()))

# Normalisation des données de test
# On s'assure qu'il s'agisse de float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# On divise par la valeur max pour obtenir un nombre adimensionné entre 0 et 1
x_train = x_train/255.0
x_test = x_test/255.0

# On transforme la matrice (28,28) en un vecteur de dimension 784 pour appliquer le clustering
X_train = x_train.reshape(len(x_train),-1)
X_test = x_test.reshape(len(x_test),-1)

# On vérifie les dimensions
print("Vérification de la dimension pour X_train, un long vecteur de taille: " + str(X_train.shape))
print("Vérification de la dimension pour X_test, un long vecteur de taille:  " + str(X_test.shape))

#On exige dans un premier temps autant de clusters que de labels (vision intuitive)
total_clusters = len(np.unique(y_test))
# On initialise notre modèle K-means
kmeans = MiniBatchKMeans(n_clusters = total_clusters, init='k-means++', batch_size=2048)
#kmedoïds= KMedoids(n_clusters = total_clusters, init='k-means++')
# On entraine notre modèle
kmeans.fit(X_train)
#kmedoïds.fit(X_train)
#On vérifie que le clustering se soit bien passé
print("Nombre de clusters: " + str(total_clusters))
print("Dimension des labels: " + str(np.shape(kmeans.labels_)))
print("Un label: " + str(kmeans.labels_[0]))


### On crée un dictionnaire qui associe à chaque cluster le label qui lui est le plus probable
def retrieve_info(cluster_labels,y_train):
    # Initialisation
    reference_labels = {}
    # Boucle: On parcourt chaque label dans la liste des labels
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i,1,0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num
    return reference_labels

#####On obtient le dictionnaire dans notre situation
reference_labels = retrieve_info(kmeans.labels_,y_train)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
  number_labels[i] = reference_labels[kmeans.labels_[i]]

# On compare la valeur prédite par le clustering et le vrai label
print("Labels prédits: " + str(number_labels[:20].astype('int')))
print("Vrais labels:   " + str(y_train[:20]))

# On calcule l'accuracy
print("Valeur de l'accuracy: " + str(accuracy_score(number_labels,y_train)))

# On calcule les différentes métriques, les différents indices de validité
def calculate_metrics(model,output):
    print('Nombre de clusters {}'.format(model.n_clusters))
    print('Inertie : {}'.format(model.inertia_))
    print('Homogénéité : {}'.format(metrics.homogeneity_score(output,model.labels_)))
    nb_clusters=model.n_clusters
    inertie=model.inertia_
    homogeneite=metrics.homogeneity_score(output,model.labels_)

    return nb_clusters,inertie,homogeneite

####On applique la méthode Elbow pour trouver K (le nombre de clusters)
cluster_number = [i for i in range (1,2)]
Acc=[]
Inertie=[]
Homogénéité=[]
for i in cluster_number:
    total_clusters = len(np.unique(y_test))
    # On initialise notre modèle
    kmeans = MiniBatchKMeans(n_clusters=i, init='k-means++', batch_size=3000)
    #kmedoid = KMedoids(n_clusters=i, init='k-means++')
    # On entraine notre modèle
    kmeans.fit(X_train)
    #kmedoid.fit(X_train)
    # On calcule les métriques
    nb_clusters, inertie, homogeneite=calculate_metrics(kmeans, y_train)
    Inertie.append(inertie)
    Homogénéité.append(homogeneite)
    # On donne les labels prédits par le modèle
    reference_labels = retrieve_info(kmeans.labels_, y_train)
    number_labels = np.random.rand(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]
    #On donne l'accuracy du modèle
    Acc.append(accuracy_score(number_labels,y_train))
    print('Accuracy: {}'.format(accuracy_score(number_labels, y_train)))
    print('\n')

fig, (ax1,ax2,ax3) = plt.subplots(3,1)
plt.subplots_adjust(hspace=0.5)
ax1.plot(cluster_number,Acc, color='C3')
ax1.set_title("Accuracy en fonction du nombre de clusters")
ax1.set_xlabel("Nombre de clusters")
ax1.set_ylabel("Accuracy")
ax2.plot(cluster_number,Inertie, color='C1')
ax2.set_title("Inertie en fonction du nombre de clusters")
ax2.set_xlabel("Nombre de clusters")
ax2.set_ylabel("Inertie")
ax3.plot(cluster_number,Homogénéité,color='b')
ax3.set_title("Homogénéité en fonction du nombre de clusters")
ax3.set_xlabel("Nombre de clusters")
ax3.set_ylabel("Homogénéité")
plt.show()


#On fait la même chose, cette fois-ci sur la base de test
# On initialise le modèle
kmeans = MiniBatchKMeans(n_clusters=256, init='k-means++', batch_size=3000)
#kmedoid= KMedoids(n_clusters=256, init='k-means++')
# On entraine le modèle
kmeans.fit(X_test)
#kmedoid.fit(X_test)
# On calcule les métriques
nb_clusters_test, inertie_test, homogeneite_test=calculate_metrics(kmeans, y_test)
# On donne les labels prédits par le modèle
reference_labels = retrieve_info(kmeans.labels_, y_test)
number_labels = np.random.rand(len(kmeans.labels_))
for i in range(len(kmeans.labels_)):
    number_labels[i] = reference_labels[kmeans.labels_[i]]
#On donne l'accuracy
print('Accuracy scores: {}'.format(accuracy_score(number_labels, y_test)))
print('\n')

####Matrice de confusion
# # On donne les labels prédits par le modèle
# reference_labels = retrieve_info(kmeans.labels_, y_train)
# number_labels = np.random.rand(len(kmeans.labels_))
# for i in range(len(kmeans.labels_)):
#     number_labels[i] = reference_labels[kmeans.labels_[i]]
print(number_labels)
print(y_test)

cm = confusion_matrix(y_test, number_labels, labels=[0,1,2,3,4,5,6,7,8,9])

print("Iteration %i %s" % (i, 70 * "_"))
#print("Label Spreading model: %d labeled & %d unlabeled (%d total)"% (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))

print(classification_report(y_test, number_labels))

print("Confusion matrix")
print(cm)

# On calcule les centroïds de notre modèle avec n_cluster=256
centroids = kmeans.cluster_centers_
print("Shape des centre" + str(centroids.shape))

centroids = centroids.reshape(256,28,28)
#On dénormalise nos données !
centroids = centroids * 255
#On trace 16 représentants de centroïds
fig, axs= plt.subplots(4,4)
plt.subplots_adjust(hspace=1)
for i in range (0,4):
    for j in range (0,4):
        axs[i,j].imshow(centroids[i*4+j])
        axs[i,j].set_title("Label prédit: " + str(reference_labels[i*4+j]))
        axs[i,j].set_xlabel("Les 28 pixels du centroïd")
        axs[i, j].set_ylabel("Les 28 pixels du centroïd")
plt.show()
