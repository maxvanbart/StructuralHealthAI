from matplotlib import pyplot as plt
from matplotlib import colors
from AE.feature_extraction import frequency_extraction
import numpy as np
import pandas as pd
import sklearn.cluster


def freq_amp_energy_plot(database, ref_amp=10**(-5), title=None):
    """Extracting frequency, amplitude and energy for clustering"""
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / ref_amp)
    full_data = pd.concat([amp_db, freq, features["energy"]], axis=1)
    data = full_data.sample(n=100000)

    plt.figure(figsize=(9, 7))
    plt.ylim(0, 1000)
    plt.xlabel("Peak amplitude of emission [dB]")
    plt.ylabel("Average frequency of emission [kHz]")
    plt.scatter(data["amplitude"], data["frequency"], c=data["energy"], s=2, norm=colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Energy [$10^{-14}$ J]')
    plt.show()


def energy_time_cluster(database, plotting=False):
    features = database
    energy, time = features["energy"], features["time"]
    data = pd.concat([energy, time], axis=1)
    labels = []
    percent = np.percentile(energy, 95)
    for i in energy:
        if i > percent:
            labels.append(1)
        else:
            labels.append(0)

    if plotting:
        plt.figure(figsize=(9, 7))
        plt.xlabel("Time [$10^{2}$ s]")
        plt.ylabel("Peak energy of emission [$10^{-14}$ J]")
        plt.scatter(data["time"]/100, data["energy"], c=labels, s=1)
        plt.show()

    return labels


def freq_amp_cluster(database, ref_amp=10**(-5),  min_samples=1500, plotting=True):
    """DBSCAN clustering"""
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / ref_amp)
    full_data = pd.concat([amp_db, freq], axis=1)
    data = full_data.sample(10000)
    init_clusters = sklearn.cluster.DBSCAN(eps=12, min_samples=min_samples).fit(data).labels_

    # old scaling method not required
    if len(set(init_clusters)) != 2:
        raise Exception(f"Unexpected number of clusters ({len(set(init_clusters))}​​​​​​) detected, try again.")
    else:
        knn_classification = sklearn.neighbors.KNeighborsClassifier(n_neighbors=100, weights='distance')
        knn_classification.fit(data, init_clusters)
        clusters = knn_classification.predict(full_data)
    full_data["clusters"] = clusters

    if plotting:
        plt.scatter(full_data['amplitude'], full_data['frequency'], c=full_data["clusters"], s=4)
        plt.xlabel("Peak amplitude of emission [dB]")
        plt.ylabel("Average frequency of emission [kHz]")
        plt.show()

    return init_clusters



