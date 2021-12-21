from matplotlib import pyplot as plt
from matplotlib import colors
from AE.feature_extraction import frequency_extraction
import numpy as np
import psutil
import pandas as pd
from AE.hit_combination import batch_split
import sklearn.cluster
import tqdm


def freq_amp_energy_plot(database, ref_amp=10**(-5), title=None):
    """Extracting frequency, amplitude and energy for clustering"""
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / ref_amp)
    full_data = pd.concat([amp_db, freq, features["energy"]], axis=1)
    data = full_data.sample(n=100000, random_state=1)

    plt.ylim(0, 1000)
    plt.figure(figsize=(9, 7))
    plt.title(title)
    plt.xlabel("Peak amplitude of emission [dB]")
    plt.ylabel("Average frequency of emission [kHz]")
    plt.scatter(data["amplitude"], data["frequency"], c=data["energy"], s=2, norm=colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Energy [$10^{-14}$ J]')
    plt.show()


def create_cluster_batches(df, delta=100, debug=False, debug_graph=False):
    """Creation of batches for clustering"""
    print("Beginning feature clustering...")
    # Find the available memory and use it to determine the maximum cluster size
    # Larger maximum clusters will avoid clusters getting split up
    available_memory = psutil.virtual_memory()[0] / 1024 ** 3
    print(f"Detected {round(available_memory,1)}GB of system memory...")
    max_size = 20000
    if available_memory > 30:
        max_size = 30000
    elif available_memory < 10:
        max_size = 10000

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


def batch_split_clst(df, max_size=20000):
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
        elif len(batches[-1]) > max_size:
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
    data = full_data.sample(n=10000, random_state=1)
    samp = 200

    """Different clustering algorithms to try"""
    """Agglomerative Clustering"""
    # clusters = sklearn.cluster.AgglomerativeClustering(n_clusters=2, compute_full_tree=True).fit(data.to_numpy())

    """DBSCAN Clustering - good for outlier detection"""
    clusters = sklearn.cluster.DBSCAN(eps=10, min_samples=samp).fit(data.to_numpy())

    """OPTICS Clustering"""
    # clusters = sklearn.cluster.OPTICS(min_samples=2).fit(data.to_numpy())

    plt.ylim(0, 1000)
    plt.figure(figsize=(9, 7))
    plt.title(f"DBSCAN Clustering with min_samples = {samp}")
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
    data = full_data.sample(n=10000, random_state=1)
    """OPTICS Clustering"""
    clusters = sklearn.cluster.OPTICS(min_samples=200).fit(data.to_numpy())

    plt.figure(figsize=(9, 7))
    plt.xlabel("Time [s]")
    plt.title(f"OPTICS Clustering with min_samples = {samp}")
    plt.xlabel("Time [$10^{-2}$ s]")
    plt.ylabel("Peak energy of emission [$10^{-14}$ J]")
    plt.scatter(data["time"] / 100, data["energy"], c=clusters.labels_, s=10)
    plt.show()

def batch_fre_amp_clst(database, ref_amp=10**(-5)):
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / ref_amp)
    full_data = pd.concat([amp_db, freq], axis=1)
    data = full_data.sample(100000)
    batches = create_cluster_batches(data)
    clusters = []
    for batch in tqdm.tqdm(batches):
        clusters.append(sklearn.cluster.DBSCAN(min_samples=4).fit(batch).labels_)
    clusters = [point for lst in clusters for point in lst]
    plt.scatter(data['amplitude'], data['frequency'], c=clusters, s=10)
    plt.show()

