def frequency_extraction(database):
    # select less features for debugging
    features = database.hits[:1000]

    # frequency extraction
    duration, counts = features["duration"], features["counts"]
    frequency = []
    for ndx in range(len(duration)):
        frequency.append(counts[ndx]/duration[ndx])
    return frequency
