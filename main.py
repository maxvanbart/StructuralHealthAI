import numpy as np
import pandas as pd

from AE.main_AE import analysis_ae
from panelObject import PanelObject


if __name__ == "__main__":
    # analysis_ae()

    objects = PanelObject.initialize_all()
    for panelobject in objects:
        print(str(panelobject.__str__()))
