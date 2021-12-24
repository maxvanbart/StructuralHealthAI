from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import time
import psutil

# old agglomerative function which is not needed anymore
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


if False:
    """Extract markers from pridb file"""
    pridb = vae.io.PriDatabase("Files/" + self.filename + "/AE/" + self.filename + ".pridb")
    markers = pridb.read_markers()
    self.start_time = markers["data"].loc[3]



def freq_amp_cluster(database, ref_amp=10**(-5)):
    """Extracting frequency, amplitude and energy for clustering"""
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / ref_amp)
    full_data = pd.concat([amp_db, freq], axis=1)
    data = full_data.sample(n=100000)

    """Different clustering algorithms to try"""
    """Agglomerative Clustering"""
    # clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=2, compute_full_tree=True).fit(data.to_numpy())

    """DBSCAN Clustering - good for outlier detection"""
    clusters = sklearn.cluster.DBSCAN(eps=10, min_samples=1700).fit(data.to_numpy())
    print(set(clusters.labels_))
    """OPTICS Clustering"""
    # clusters = sklearn.cluster.OPTICS(min_samples=2).fit(data.to_numpy())

    plt.ylim(0, 1000)
    plt.title(f"DBSCAN Clustering with min_samples = 1700")
    plt.xlabel("Peak amplitude of emission [dB]")
    plt.ylabel("Average frequency of emission [kHz]")
    plt.scatter(data["amplitude"], data["frequency"], c=clusters.labels_, s=4)
    plt.show()


def all_features_cluster(database, ref_amp=10**(-5)):
    """Extract all features for clustering, and convert ampltiude to dB"""
    features = database
    features["amplitude"] = 20 * np.log10(features["amplitude"] / ref_amp)
    full_data = pd.concat([features, frequency_extraction(features).divide(1000)], axis=1)
    data = full_data.sample(n=10000, random_state=1)

    """Different clustering algorithms to try"""
    """Agglomerative Clustering"""
    # clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=2, compute_full_tree=True).fit(data.to_numpy())

    """DBSCAN Clustering"""
    # clusters = sklearn.cluster.DBSCAN(eps=5, min_samples=2).fit(data.to_numpy())

    """OPTICS Clustering - only one that gives interpretable results"""
    clusters = sklearn.cluster.OPTICS(min_samples=4).fit(data.to_numpy())

    plt.ylim(0, 1000)
    plt.figure(figsize=(9, 7))
    plt.xlabel("Peak amplitude of emission [dB]")
    plt.ylabel("Average frequency of emission [kHz]")
    plt.scatter(data["amplitude"], data["frequency"], c=clusters.labels_, s=4)
    plt.show()


def create_cluster_batches(df, delta=100, debug=False, debug_graph=False, max_size=10000):
    """Creation of batches for clustering"""
    print("Beginning feature clustering...")
    # Find the available memory and use it to determine the maximum cluster size
    # Larger maximum clusters will avoid clusters getting split up
    available_memory = psutil.virtual_memory()[0] / 1024 ** 3
    print(f"Detected {round(available_memory,1)}GB of system memory...")
    '''max_size = 20000
    if available_memory > 30:
        max_size = 30000
    elif available_memory < 10:
        max_size = 10000
    '''
    batches = batch_split_clst(df, max_size=max_size)

    # print some information about the batches if debug is enabled
    if debug:
        print(len(batches))
        for batch in batches:
            print(batch.shape)

    # Enabling this debug graph will show the batch division of the selected datapoints
    if debug_graph:
        n = 0
        for batch in batches:
            n += len(batch)
            plt.figure(figsize=(9, 7))
            plt.scatter(batch[:, 0], batch[:, 4], s=4)
        print(n)
        plt.xlabel("Time")
        plt.ylabel("RMS voltage")
        plt.show()

    return batches


def batch_split_clst(df, max_size):
    """Batch splitting fro clustering using only a maximum size"""
    if type(df) == pd.core.frame.DataFrame:
        matrix = df.values.tolist()
    else:
        matrix = df
    batches = []
    for row in matrix:
        # insert the first row if the batches list is empty
        if len(batches) == 0:
            batches.append([row])
        # after inserting the first row we use the max_size to decide if we put it in the latest batch
        # or if we should start a new batch
        elif len(batches[-1]) >= max_size:
            batches.append([row])
        else:
            batches[-1].append(row)
    return batches

