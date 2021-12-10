import luna_data_to_python
import numpy as np
import matplotlib.pyplot as plt

array, df = luna_data_to_python.raw_to_python("Files/L1-03/LUNA/", 'L1-03.txt')
print(df.head(1))
row2 = array[1:4, 1:10]
print(row2)
timestamp = array[:, 0]

x = np.linspace(0, 18, 9)
y = np.linspace(0, 6, 3)
print(row2.shape)
print(x, y)
print(type(row2[0, 0]))

fig, ax = plt.subplots()
ax.pcolormesh(x, y, row2)

# plt
# plt.imshow(row2)
plt.show()
