# StructuralHealthAI

- It is assumed all future data uses the same data format which has been used previously,
  if this is not the case either the code, or the data layout has to be changed.
- Has been successfully tested on Windows, unknown if compatible with other operating systems.
- The user has an option to skip (re)generating the clustered databases to save time, in case of multiple reruns.
- The user can also enable a plot of the final clustering to aid visualization of the clustered data.
- The final product will contain CSV files with all the clustered databases per sensor and an image of the clustered data.

# Todo:

- Folder structure
- Extra modules used

## LUNA

- The LUNA sensor file contains the start and end values of where the sensor is attached to the panel, 
  if a new panel is investigated with this program the new start and end values must be added to this file.
- To properly synchronise the LUNA data with the AE data the LUNA data is preprocessed to remove outliers, 
  this part of the code has been tuned to work with the provided panels, but does not guarantee support for future panels.
  If the setup of the experiment in terms of measurement intervals changes significantly from the default case, this part of
  the code has to be checked thoroughly.
-   

## AE

## PZT

NOTE TO SELF: if you want to redefine the threshold, force regenerating the databases