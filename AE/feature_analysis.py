from matplotlib import pyplot as plt
from matplotlib import colors
from AE.feature_extraction import frequency_extraction
import numpy as np
import psutil
import pandas as pd
from AE.hit_combination import batch_split
import sklearn.cluster
from tqdm import tqdm


def freq_amp_energy_plot(database, ref_amp=10**(-5), title=None):
    """Extracting frequency, amplitude and energy for clustering"""
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / ref_amp)
    full_data = pd.concat([amp_db, freq, features["energy"]], axis=1)
    data = full_data.sample(n=100000)

    plt.figure(figsize=(9, 7))
    plt.ylim(0, 1000)
    plt.title(title)
    plt.xlabel("Peak amplitude of emission [dB]")
    plt.ylabel("Average frequency of emission [kHz]")
    plt.scatter(data["amplitude"], data["frequency"], c=data["energy"], s=2, norm=colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Energy [$10^{-14}$ J]')
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


def freq_amp_cluster(database, ref_amp=10**(-5)):
    """Extracting frequency, amplitude and energy for clustering"""
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / ref_amp)
    full_data = pd.concat([amp_db, freq], axis=1)
    data = full_data.sample(n=30000)

    """Different clustering algorithms to try"""
    """Agglomerative Clustering"""
    # clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=2, compute_full_tree=True).fit(data.to_numpy())

    """DBSCAN Clustering - good for outlier detection"""
    clusters = sklearn.cluster.DBSCAN(eps=10, min_samples=150).fit(data.to_numpy())
    print(set(clusters.labels_))
    """OPTICS Clustering"""
    # clusters = sklearn.cluster.OPTICS(min_samples=2).fit(data.to_numpy())

    plt.ylim(0, 1000)
    plt.title(f"DBSCAN Clustering with min_samples = 150")
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


def freq_amp_time_cluster(database, ref_amp=10**(-5)):
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / ref_amp)
    ndx = np.random.randint(0, len(amp), 10000)
    plt.ylim(0, 1000)
    plt.figure(figsize=(9, 7))
    plt.xlabel("Peak amplitude of emission [dB]")
    plt.ylabel("Average frequency of emission [kHz]")
    plt.scatter(amp_db.loc[ndx], freq.loc[ndx], s=1, c=features["time"].loc[ndx], norm=colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Time [s]')
    plt.show()


def energy_time_cluster(database):
    features = database
    energy, time = features["energy"], features["time"]
    full_data = pd.concat([energy, time], axis=1)
    data = full_data.sample(n=10000)
    samp = 150
    """OPTICS Clustering"""
    clusters = sklearn.cluster.KMeans(n_clusters=20).fit(data["energy"].to_numpy().reshape(-1, 1))

    plt.figure(figsize=(9, 7))
    plt.xlabel("Time [s]")
    plt.title(f"OPTICS Clustering with min_samples = {samp}")
    plt.xlabel("Time [$10^{-2}$ s]")
    plt.ylabel("Peak energy of emission [$10^{-14}$ J]")
    plt.scatter(data["time"] / 100, data["energy"], c=clusters.labels_, s=10)
    plt.show()


def batch_fre_amp_clst(database, ref_amp=10**(-5),  min_samples=150):
    """DBSCAN clustering"""
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / ref_amp)
    full_data = pd.concat([amp_db, freq], axis=1)
    full_data = full_data.sample(1000000)
    data = full_data.sample(10000)
    init_clusters = sklearn.cluster.DBSCAN(eps=10, min_samples=min_samples).fit(data).labels_

    if len(set(init_clusters)) != 2:
        raise Exception(f"Unexpected number of clusters ({set(init_clusters)} detected, try again.")
    else:
        knn_classification = sklearn.neighbors.KNeighborsClassifier(n_neighbors=30)
        knn_classification.fit(data, init_clusters)
        clusters = knn_classification.predict(full_data)

    full_data["clusters"] = clusters
    plt.scatter(full_data['amplitude'], full_data['frequency'], c=full_data["clusters"], s=4)
    plt.show()

    # batches = create_cluster_batches(data, max_size=12000)
    # clusters = []
    # for batch in tqdm.tqdm(batches):
    #     new_cluster = sklearn.cluster.DBSCAN(eps=10, min_samples=min_samples).fit(batch).labels_
    #     if len(set(new_cluster)) > 2:
    #         print(set(new_cluster))
    #         clusters.append(new_cluster)
    #         # raise Exception("More than two clusters, unexpected result. Please change the max_size")
    #     else:
    #         clusters.append(new_cluster)
    #
    # clusters = [point for lst in clusters for point in lst]
    # data["clusters"] = clusters
    # # data = data.loc[(data['clusters'] != 0) & (data['clusters'] != 1)]
    # plt.scatter(data['amplitude'], data['frequency'], c=data["clusters"], s=4)
    # plt.show()


def batch_eny_time_clst(database):
    features = database
    energy, time = features["energy"], features["time"]
    full_data = pd.concat([energy, time], axis=1)
    data = full_data.sample(1000000)
    batches = create_cluster_batches(data)
    clusters = []
    for batch in tqdm.tqdm(batches):
        clusters.append(sklearn.cluster.KMeans(n_clusters=15).fit(np.array(batch)[:, 0].reshape(-1, 1)).labels_)
    clusters = [point for lst in clusters for point in lst]
    plt.scatter(data['time'], data['energy'], c=clusters, s=10)
    plt.show()
