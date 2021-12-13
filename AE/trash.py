from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import time


def agglomerative(X, time_lst, rms, features):
    # Time complexity O(n2) and high memory complexity
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.01, compute_full_tree=True)
    t0 = time.time()
    clustering = model.fit(X)
    t1 = time.time()
    print(f"Time elapsed while clustering: {round(t1 - t0, 3)} seconds")

    labels = clustering.labels_
    # print(labels)
    clusterdict = {}
    for n in labels:
        if n in clusterdict:
            clusterdict[n] += 1
        else:
            clusterdict[n] = 1

    print(f"Amount of clusters: {len(clusterdict)}")
    plt.scatter(time_lst, rms, s=4, c=clustering.labels_)
    plt.xlabel("Time")
    plt.ylabel("RMS voltage")
    for i, label in enumerate(labels):
        plt.annotate(f"{label}", (np.array(features["time"])[i], np.array(features["rms"])[i]))

    plt.show()
