import numpy as np
import pandas as pd

from AE.main_AE import analysis_ae
from panelObject import Panel


if __name__ == "__main__":
    # analysis_ae()

    panels = Panel.initialize_all()
    for panel in panels:
        print(str(panel.__str__()))
        panel.load_ae()
