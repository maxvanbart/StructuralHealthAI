import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from AE.feature_extraction import frequency_extraction
import matplotlib.pyplot as plt
from AE.hit_combination import batch_split
import psutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


def clustering(database):
    frequency = frequency_extraction(database)
    amp = database['amplitude']
    amp_db = 20 * np.log10(amp / (10 ** (-5)))
    full_data = pd.concat([amp_db, frequency], axis=1)
    data = full_data.sample(n=10000, random_state=1)
    X = data.to_numpy()
    '''X = pd.DataFrame(X)
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(X)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    plt.figure(figsize=(8, 8))
    plt.title('Visualising the data')
    Dendrogram = shc.dendrogram((shc.linkage(X_principal, method='ward')))'''



    kmeans = AgglomerativeClustering(n_clusters=6, compute_full_tree=True).fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.xlabel('amp')
    plt.ylabel('frequency')
    plt.show()
