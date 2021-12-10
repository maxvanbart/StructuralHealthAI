from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def init_clustering(database):

    features = database.hits
    features = database.hits[database.hits["channel"] == 1]
    features = features[:100]
    time, rms = features[["time"]], features["rms"]
    plt.scatter(time, rms, s=4, c=features["channel"])
    plt.xlabel("Time")
    plt.ylabel("RMS voltage")
    plt.show()
    X = np.transpose(np.array([features["time"], features["rms"]]))
    agglomerative(X, time, rms, features)


def agglomerative(X, time, rms, features):
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.01, compute_full_tree=True)
    clustering = model.fit(X)
    labels = clustering.labels_
    print(labels)
    clusterdict = {}
    for n in labels:
        if n in clusterdict:
            clusterdict[n] += 1
        else:
            clusterdict[n] = 1

    print(f"Amount of clusters: {len(clusterdict)}")
    plt.scatter(time, rms, s=4, c=clustering.labels_)
    plt.xlabel("Time")
    plt.ylabel("RMS voltage")
    for i, label in enumerate(labels):
        plt.annotate(label, (np.array(features["time"])[i], np.array(features["rms"])[i]))

    plt.show()
