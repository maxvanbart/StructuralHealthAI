from matplotlib import pyplot as plt

def init_clustering(database):
    features = database.hits[database.hits["channel"] == 1]
    features = features[:1000]
    time, rms = features[["time"]], features["rms"]
    plt.scatter(time, rms, s=4, c=features[["channel"]])
    plt.xlabel("Time")
    plt.ylabel("RMS voltage")
    plt.show()
    # hi
