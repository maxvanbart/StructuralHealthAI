import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from AE.feature_extraction import frequency_extraction
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from matplotlib import colors
from AE.hit_combination import batch_split
import psutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from utilities.cluster_scoring import calinski_score, silhouette_score, davies_score


def clustering(database):
    frequency = frequency_extraction(database)
    amp = database['amplitude']
    amp_db = 20 * np.log10(amp / (10 ** (-5)))
    time = database['time']
    energy = database['energy']
    rst = database['rise_time']
    full_data = pd.concat([amp_db,frequency, time, energy, rst], axis=1)
    data = full_data.sample(n=10000, random_state=1)
    X = data.to_numpy()
    '''k = np.arange(2,20)
    s = []
    c = []
    d = []
    for i in k:
        ac = AgglomerativeClustering(n_clusters=i, compute_full_tree=True).fit(X)
        s.append(silhouette_score(X ,ac.labels_))
        c.append(calinski_score(X, ac.labels_))
        d.append(davies_score(X, ac.labels_))
    lst = [s, c, d]
    for i in lst:
        plt.bar(k, i)
        plt.xlabel('Number of clusters', fontsize=20)
        plt.ylabel('S(i)', fontsize=20)
        plt.show()'''

    ac3 = OPTICS(min_samples=200).fit(X[:,(2,3)])
    '''ax = plt.axes(projection='3d')
    ax.scatter(X[:,3],X[:,2],X[:,1], c=ac3.labels_, cmap='viridis', linewidth=0.5)
    ax.set_xlabel('energy')
    ax.set_ylabel('time')
    ax.set_zlabel('freq')'''

    plt.scatter(X[:, 2],X[:,3], c=ac3.labels_)
    cbar = plt.colorbar()
    cbar.set_label('freq')
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.show()


def clustering_time_energy(database):
    energy = database['energy']
    time = database['time']
    full_data = pd.concat([time, energy], axis=1)
    data = full_data.sample(n=10000, random_state=1)
    X = data.to_numpy()

    ac3 = OPTICS(min_samples=200).fit(X[:,(0,1)])

    plt.scatter(X[:, 0],X[:,1], c=ac3.labels_)
    plt.xlabel('time')
    plt.ylabel('energy')
    plt.show()
