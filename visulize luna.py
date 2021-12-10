import luna_data_to_python
import numpy as np
import matplotlib.pyplot as plt

array, df = luna_data_to_python.raw_to_array("Files/L1-03/LUNA/", 'L1-03.txt')
print(df.head(1))
row2 = array[1:4, 1:30]
print(row2)
timestamp = array[:, 0]

x = np.linspace(0, 18, 9)
y = np.linspace(0, 6, 3)
print(row2.shape)
print(x, y)
print(type(row2[0, 0]))

# fig, ax = plt.subplots()
# ax.pcolormesh(x, y, row2)

# plt
# plt.imshow(row2)
plt.show()


def print_line_of_array(array, row_numbs, difference=False, color1="g", color2="r"):
    """
    array = np.array
    row_numbs is a list, could be single value list
    difference is to plot absolute difference between two rows. Only works if row_numbs is 2 value list
    color1 is color, dafault green
    color2 is color, default red
    """
    if difference is True and len(row_numbs)!= 2:
        print("Function Error: Change difference to False or use a 2 value list")
        return

    while difference and len(row_numbs) == 2:
        diffrow = []
        row1 = array[row_numbs[0]]
        row2 = array[row_numbs[1]]
        for j in range(len(row1)):
            diffrow.append(abs(row1[j]) - abs(row2[j])) # absolute difference
        break

    for row_numb in row_numbs:
        row = array[row_numb]
        # row is a row from array with length
        rowlengt = len(row)
        avg_row_value = np.nansum(row)/rowlengt
        maxvalue, minvalue = np.nanmax(row), np.nanmin(row)
        print(avg_row_value, maxvalue, minvalue)
        if difference is False:
            plt.plot(row, "o-", label=f"row {row_numb+1}")
            plt.title("Lengt x Micro strain")
            plt.xlabel("Length")
            plt.ylabel("Micro strain")
            plt.legend()

    if difference is True:
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        ax1.plot(row1, f"{color1}o-", label=f"row {row_numb+1}")
        ax1.set_title(f"array numb {row_numbs[0]}")
        ax2.plot(row2, f"{color1}o-", label=f"row {row_numb+1}")
        ax2.set_title(f"array numb {row_numbs[1]}")
        ax3.plot(diffrow, f'{color2}')
        ax3.set_title(f"absolute difference")
    plt.show()


print_line_of_array(row2, [0, 1], difference=True)

green = [0, 255, 0]
red = [255, 0, 0]
purple = [128, 0, 128]
white = [255, 255, 255]
black = [0, 0, 0]


def colorscale(array):
    output = []
    maxpeak = 0
    minpeak = 100

    # get maximal value and lowest value
    for number, row in enumerate(array):
        for item in row:
            if item > maxpeak:
                maxpeak = item
            if item < minpeak:
                minpeak = item


    for number, row in enumerate(array):
        for item in row:
            if item >= 0: # positive value
                output.append([0, item/maxpeak*255, 0])
            if item < 0: # negative value
                output.append([item/minpeak*255, 0, 0])
            if item == np.nan: # NaN
                output.append(purple)