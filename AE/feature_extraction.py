import pandas as pd


def frequency_extraction(hits):
    return pd.DataFrame(hits["counts"] / hits["duration"], columns=['frequency'])
