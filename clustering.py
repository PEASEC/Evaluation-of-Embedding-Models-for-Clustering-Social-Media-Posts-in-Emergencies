from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from spherecluster import SphericalKMeans
from spherecluster import VonMisesFisherMixture
import numpy as np

def cluster(documents, cluster_count, cluster_algorithm):
    clusterer = globals()[cluster_algorithm](n_clusters=cluster_count).fit(np.array(documents))
    return clusterer.labels_, clusterer.cluster_centers_

def get_k_nearest(documents, cluster_center, k):
    tree = spatial.KDTree(documents)
    distances, indices = tree.query(cluster_center, k)
    return distances, indices

def evaluate(documents, labels):
    silhouette = silhouette_score(np.array(documents), labels=labels)
    calinski_harabasz = calinski_harabasz_score(np.array(documents), labels=labels)
    davies_bouldin = davies_bouldin_score(np.array(documents), labels=labels)

    print("####################### Evaluation #######################")
    print("# Silhouette Score:", str(silhouette))
    print("# Calinski Harabasz Score:", str(calinski_harabasz))
    print("# Davies Bouldin Score:", str(davies_bouldin))
    print("##########################################################")
    return silhouette, calinski_harabasz, davies_bouldin
