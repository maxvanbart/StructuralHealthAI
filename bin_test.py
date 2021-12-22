import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# Some variables to mimic the PanelObject class
name = 'L1-09'
location = 'Files/' + name + "/AE/" + name + "-clustered.csv"
bin_width = 1

# Load the clustered database and sort it by time
ae_clustered_database = pd.read_csv(location)
df = ae_clustered_database.sort_values(by=['time'])
array = np.array(df)

# Determine the highest value of t and use it to create the bins which will be used
final_time = df[df['time'] == df['time'].max()]
final_time = int(np.ceil(np.array(final_time['time'])[0])+1)
print("Final value of t:", final_time)

bin_count = int(np.ceil(final_time/bin_width))
bins = [0] * bin_count
trange = range(0, bin_count)

for row in list(array):
    x = row[0]
    y = int(x//bin_width)
    bins[y] += 1
print("Finished counting")

plt.bar(trange, bins, width=1)
plt.title(name)
plt.xlabel("Bin")
plt.ylabel("Datapoint density")

plt.show()
