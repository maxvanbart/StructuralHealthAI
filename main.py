import time
from tqdm import tqdm
from panelObject import Panel
import os


def main(force_clustering=False, visualization=False, pzt_thr=0.1):
    # start time
    t0 = time.time()

    # initialize all the panels from the folders
    panels = Panel.initialize_all(pzt_thr, force_clustering=force_clustering, plotting=visualization)

    # select specific panels
    # panels = panels[:1]

    # for every panel we perform the following actions
    for panel in tqdm(panels, desc='Panel'):
        print('\n' + str(panel))
        # Prepare AE.
        panel.load_ae()
        panel.analyse_ae()

        # Prepare LUNA.
        panel.load_luna()
        panel.synchronise_luna()
        panel.analyse_luna()

        # Prepare PZT.
        panel.load_pzt()
        panel.synchronise_pzt()
        panel.analyse_pzt()

        # Plot and save all the clusters.
        panel.save_all()
        panel.visualize_all(visualization)

    # end time, it also prints the elapsed time
    t1 = time.time()
    print(f"Total time elapsed: {round((t1 - t0) / 60, 3)} minutes")


# This line is needed for implementing multi-threading
if __name__ == "__main__":
    print("")
    fc = ""
    vs = ""
    # Ask to regenerate files
    while fc not in ["y", "n"]:
        fc = input("Do you want to regenerate all databases? (y/n): ").lower()
        if fc not in ["y", "n"]:
            print("Please respond with yes (y) or no (n)")
    if fc == "y":
        fc = True
    else:
        fc = False

    # Ask whether to enable visualization
    while vs not in ["y", "n"]:
        vs = input("Do you want to enable visualization? (y/n): ").lower()
        if vs not in ["y", "n"]:
            print("Please respond with yes (y) or no (n)")
    if vs == "y":
        vs = True
    else:
        vs = False

    # Ask for a custom PZT threshold
    if fc:
        succes = False
        print("Do you want to use a custom threshold for the PZT data?")
        print("Leave empty for default value of 0.1 or enter value between 0 and 1.")
        print("Please note that databases need to be regenerated for a different threshold value to take effect.")
        while not succes:
            thr = input("Please enter PZT threshold value: ")
            try:
                thr = float(thr)
                if 1 > thr > 0:
                    succes = True
            except ValueError:
                if thr == "":
                    succes = True
                    thr = 0.1
            if not succes:
                print("Please enter a valid value or leave empty for default")
        print(f"Using a threshold value of {thr} for the PZT data.")
    else:
        thr = 0.1
    main(force_clustering=fc, visualization=vs, pzt_thr=thr)
