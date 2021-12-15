import pandas as pd


def frequency_extraction(hits):
    # frequency extraction
    duration, counts = hits["duration"], hits["counts"]
    frequency = pd.DataFrame(counts/duration)
    return frequency
