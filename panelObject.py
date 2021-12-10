import os


from AE.utilities import Pridb


files_folder = "Files"


class PanelObject:
    def __init__(self, name):
        self.dir_name = name

    @staticmethod
    def initialize_all():
        panels = os.listdir(files_folder)
        lst = []
        for panel in panels:
            lst.append(PanelObject(panel))

        return lst

    def __repr__(self):
        return f"PanelObject({self.dir_name})"

    def __str__(self):
        return f"Panel {self.dir_name}"
