import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Ribbon:
    def __init__(self, bw):
        self.t_lst = []
        self.bin_width = bw
        self.point_count = 0

        self.t_start = None
        self.t_end = None
        self.width = None

    def add_entry(self, j, m):
        self.t_lst.append(j)
        self.point_count += m

    def update(self):
        self.t_start = self.t_lst[0]
        self.t_end = self.t_lst[-1]
        self.width = (self.t_end - self.t_start + 1)*self.bin_width

    def __str__(self):
        self.update()
        points = self.point_count
        return f"Ribbon of width {self.width} containing {points} points"

    def __repr__(self):
        return f"Ribbon({self.bin_width})"


class Entry:
    def __init__(self):
        self.name = 'Hans'


# Some variables to mimic the PanelObject class
name = 'L1-03'
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
print("Finished counting...")

ribbon_lst = []
found_prev = False
not_found_count = 99
for i in trange:
    if bins[i] > 0:
        not_found_count = 0
        if found_prev:
            ribbon_lst[-1].add_entry(i, bins[i])
        else:
            ribbon_lst.append(Ribbon(bin_width))
            ribbon_lst[-1].add_entry(i, bins[i])
            found_prev = True
    else:
        if not_found_count > 0:
            found_prev = False
        else:
            not_found_count += 1

print(f"Generated {len(ribbon_lst)} ribbons...")
for ribbon in ribbon_lst:
    ribbon.update()

ribbon_lst = [x for x in ribbon_lst if x.width > 3]

ribbon_width_lst = [x.width for x in ribbon_lst]
while np.std(ribbon_width_lst) > 15:
    mean_width = np.mean(ribbon_width_lst)
    ribbon_lst = [x for x in ribbon_lst if x.width > mean_width]
    ribbon_width_lst = [x.width for x in ribbon_lst]

print(np.mean(ribbon_width_lst))
print(np.std(ribbon_width_lst))
print(len(ribbon_lst))

for ribbon in ribbon_lst:
    print(str(ribbon))

# plt.bar(trange, bins, width=1)
# plt.title(name)
# plt.xlabel("Bin")
# plt.ylabel("Datapoint density")
#
# plt.show()
