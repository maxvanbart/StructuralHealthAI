import pandas as pd


def frequency_extraction(hits):
    return pd.DataFrame(hits["duration"] / hits["counts"], columns=['frequency'])
