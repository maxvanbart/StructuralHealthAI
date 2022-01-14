import numpy as np


class Ribbon:
    """This class can store many variables related to the ribbons observed in the AE data"""
    def __init__(self, bw):
        self.t_lst = []
        self.bin_width = bw
        self.point_count = 0

        self.t_start = None
        self.t_end = None
        self.width = None

    def add_entry(self, j, m):
        """This function can be used to add the contents of a bin to the ribbon"""
        self.t_lst.append(j)
        self.point_count += m

    def update(self):
        """This function can be used to calculate some interesting facts about the ribbon"""
        # self.t_start = self.t_lst[0]
        self.t_start = min(self.t_lst)
        # self.t_end = self.t_lst[-1]
        self.t_end = max(self.t_lst)
        self.width = (self.t_end - self.t_start + 1)*self.bin_width

    def __str__(self):
        self.update()
        points = self.point_count
        return f"Ribbon of width {self.width} containing {points} points"

    def __repr__(self):
        return f"Ribbon({self.bin_width})"
