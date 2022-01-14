import time
from tqdm import tqdm
from panelObject import Panel


def main(force_clustering=False, visualization=False):
    # start time
    t0 = time.time()

    # initialize all the panels from the folders
    panels = Panel.initialize_all(force_clustering=force_clustering, plotting=visualization)

    # select specific panels
    panels = panels[:1]

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

        # Do PZT stuff
        panel.load_pzt()
        panel.synchronise_pzt()
        panel.analyse_pzt()

        # Plot and save all the clusters.
        if visualization:
            panel.visualize_all()
        panel.save_all()

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
    main(force_clustering=fc, visualization=vs)
