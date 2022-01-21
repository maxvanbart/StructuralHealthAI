# # # Structural Health Monitoring - Capstone AI # # #

# # READ BEFORE RUNNING # #
- To load the measurements from experiments into this program, a complete .zip for all panels must be downloaded from 
  Dataverse (https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/QNURER). Then, the complete .zip must be put 
  into the Files folder (see Folder Structure for more information). Then, the user must select "Extract here" on it and 
  repeat this for every single .zip folder (e.g. AE.zip in L1-03) that it contains.
- In Files > (Panel Name), a new folder named Results will be created on first run. Here, the final databases and 
  any visualisations or plots will be stored.


# # General Information / Running for the first time # #
- To run this program, main.py must be executed. It will prompt some inputs from the user, such as forcibly generating
  databases or enabling visualisations.
- In case of multiple re-runs, the code will not regenerate clustered databases and instead load them from saved
  .csv files in the Results folder. It is possible to forcibly regenerate all databases if desired.
- The user can enable visualisation within Python (pop-up plots). If this is disabled, the plots will still be saved in
  the Results folder.
- This program has been successfully tested on Windows, unknown if compatible with other operating systems.


# # Required Libraries # #
- tqdm              - sklearn
- numpy             - psutil
- matplotlib
- pandas
- scipy


# # Folder Structure # #
Project Folder
│   README.md
│   main.py
│   panelObject.py
└───Files
│   │   Impact_Locations.pdf
│   │   MANIFEST.TXT
│   └───L1-03
│       │   L1-03.pdf
│       └───AE (if any of these folders are a .zip, they must be unzipped.)
│       └───LUNA
│       └───PZT
│       └───Results (created on first run of the program)
│   └───l1-04
│   └───L1-05 
|   └───...
└───AE
│   │   clustering.py
│   │   feature_analysis.py
│   │   feature_extraction.py
│   │   hit_combination.py
│   │   utilities.py
└───LUNA
│   │   luna_array_to_cluster.py
│   │   luna_data_to_array.py
│   │   luna_postprocessing.py
│   │   luna_preprocessing.py
│   │   LUNA_sensor.txt
└───PZT
│   │   analyze_pzt.py
│   │   feature_extractor.py
│   │   load_pzt.py
└───TimeSync
│   │   dataTypeClasses.py
│   │   ribbonFinder.py
│   │   timeSync.py
│   │   translateLuna.py
│   │   translatePZT.py
└───utilities
│   │   datetime_conv.py


# # LUNA # #
- The LUNA sensor file contains the start and end values of where the sensor is attached to the panel, 
  if a new panel is investigated with this program the new start and end values must be added to this file.
- To properly synchronise the LUNA data with the AE data the LUNA data is preprocessed to remove outliers, 
  this part of the code has been tuned to work with the provided panels, but does not guarantee support for future panels.
  If the setup of the experiment in terms of measurement intervals changes significantly from the default case, this part of
  the code has to be checked thoroughly.


# # AE # #
- The AE_panel name.csv contains a database with only high-energy events and their extracted features.
- All other measurements are synchronised in time relative to the start of the AE measurements.

# # PZT # #
- PZT database has averaged the 10 measurements per frequency. It lists the emitted frequency, as well as the state
  number with the corresponding time relative to the AE data for each measurement.
- It is possible to redefine the threshold that is used to calculate e.g. risetime. This is done by forcing the databases    
  to regenerate, at which point you will be asked to input a new threshold (default 0.1)
