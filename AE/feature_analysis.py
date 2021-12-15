from matplotlib import pyplot as plt
from matplotlib import colors
from AE.feature_extraction import frequency_extraction
import numpy as np


def freq_amp_energy_cluster(database):
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / (10 ** (-5)))
    ndx = np.random.randint(0, len(amp), 100000)
    plt.ylim(0, 1000)
    plt.xlabel("Amplitude [dB]")
    plt.ylabel("Frequency [kHz]")
    plt.scatter(amp_db.loc[ndx], freq.loc[ndx], s=1, c=features["energy"].loc[ndx], norm=colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Energy [eu]')
    plt.show()


def freq_amp_time_cluster(database):
    features = database
    amp, freq = features["amplitude"], frequency_extraction(features).divide(1000)
    amp_db = 20 * np.log10(amp / (10 ** (-5)))
    ndx = np.random.randint(0, len(amp), 100000)
    plt.ylim(0, 1000)
    plt.xlabel("Amplitude [dB]")
    plt.ylabel("Frequency [kHz]")
    plt.scatter(amp_db.loc[ndx], freq.loc[ndx], s=1, c=features["time"].loc[ndx], norm=colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Time [s]')
    plt.show()
